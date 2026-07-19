#include "server.h"

#include <QtConcurrent>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonValue>
#include <QMap>
#include <QMutexLocker>
#include <QPointer>
#include <QRegularExpression>
#include <QTcpSocket>
#include <QTextStream>
#include <QUrlQuery>
#include <QUuid>
#include <cmath>
#include <future>
#include <unordered_map>
#include <windows.h>
#include <winhttp.h>

namespace {
QString envValue(const QString& name, const QString& fallback = {}) {
  const auto value = qEnvironmentVariable(name.toUtf8().constData());
  return value.isEmpty() ? fallback : value;
}

QMap<QString, QString> readEnvFile(const QString& path) {
  QMap<QString, QString> values;
  QFile file(path);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return values;
  while (!file.atEnd()) {
    const QString line = QString::fromUtf8(file.readLine()).trimmed();
    if (line.isEmpty() || line.startsWith('#')) continue;
    const int equals = line.indexOf('=');
    if (equals > 0) values[line.left(equals).trimmed()] = line.mid(equals + 1).trimmed();
  }
  return values;
}

QString fromWide(const std::wstring& value) { return QString::fromWCharArray(value.data(), static_cast<int>(value.size())); }

struct WebResult { int status{0}; QByteArray body; QString error; };

WebResult webRequest(const QString& method, const QString& rawUrl, const QByteArray& body = {}, const QMap<QString, QString>& headers = {}) {
  const QUrl url(rawUrl);
  if (!url.isValid()) return {0, {}, "invalid URL"};
  HINTERNET session = WinHttpOpen(L"AIInfraQA/1.0", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!session) return {0, {}, "WinHttpOpen failed"};
  WinHttpSetTimeouts(session, 10000, 10000, 120000, 120000);
  const std::wstring host = url.host().toStdWString();
  const INTERNET_PORT port = static_cast<INTERNET_PORT>(url.port(url.scheme() == "https" ? 443 : 80));
  HINTERNET connection = WinHttpConnect(session, host.c_str(), port, 0);
  if (!connection) { WinHttpCloseHandle(session); return {0, {}, "WinHttpConnect failed"}; }
  QString path = url.path(QUrl::FullyEncoded); if (path.isEmpty()) path = "/";
  if (!url.query().isEmpty()) path += "?" + url.query(QUrl::FullyEncoded);
  const std::wstring pathWide = path.toStdWString(); const std::wstring methodWide = method.toStdWString();
  const DWORD flags = url.scheme() == "https" ? WINHTTP_FLAG_SECURE : 0;
  HINTERNET request = WinHttpOpenRequest(connection, methodWide.c_str(), pathWide.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
  if (!request) { WinHttpCloseHandle(connection); WinHttpCloseHandle(session); return {0, {}, "WinHttpOpenRequest failed"}; }
  QString headerText = "Content-Type: application/json\r\n";
  for (auto it = headers.begin(); it != headers.end(); ++it) headerText += it.key() + ": " + it.value() + "\r\n";
  const std::wstring headerWide = headerText.toStdWString();
  const BOOL sent = WinHttpSendRequest(request, headerWide.c_str(), static_cast<DWORD>(headerWide.size()), body.isEmpty() ? WINHTTP_NO_REQUEST_DATA : const_cast<char*>(body.data()), static_cast<DWORD>(body.size()), static_cast<DWORD>(body.size()), 0);
  if (!sent || !WinHttpReceiveResponse(request, nullptr)) { WinHttpCloseHandle(request); WinHttpCloseHandle(connection); WinHttpCloseHandle(session); return {0, {}, QString("HTTP request failed: %1").arg(GetLastError())}; }
  DWORD status = 0, statusSize = sizeof(status); WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &statusSize, WINHTTP_NO_HEADER_INDEX);
  QByteArray response;
  while (true) { DWORD available = 0; if (!WinHttpQueryDataAvailable(request, &available) || available == 0) break; QByteArray chunk(static_cast<int>(available), Qt::Uninitialized); DWORD read = 0; if (!WinHttpReadData(request, chunk.data(), available, &read)) break; chunk.resize(static_cast<int>(read)); response += chunk; }
  WinHttpCloseHandle(request); WinHttpCloseHandle(connection); WinHttpCloseHandle(session);
  return {static_cast<int>(status), response, {}};
}

QJsonObject errorResponse(const QString& message) { return QJsonObject{{"detail", message}}; }

QString statusText(int status) {
  if (status == 200) return "OK"; if (status == 201) return "Created"; if (status == 400) return "Bad Request";
  if (status == 401) return "Unauthorized"; if (status == 404) return "Not Found"; if (status == 503) return "Service Unavailable";
  return "Error";
}

QString stripFence(QString text) {
  text = text.trimmed();
  if (text.startsWith("```")) { const int newline = text.indexOf('\n'); if (newline >= 0) text = text.mid(newline + 1); if (text.endsWith("```")) text.chop(3); }
  return text.trimmed() + "\n";
}
}

BackendServer::BackendServer(QObject* parent) : QObject(parent) {
  connect(&server_, &QTcpServer::newConnection, this, &BackendServer::acceptConnection);
  const auto env = readEnvFile(QDir::current().filePath(".env"));
  const auto get = [&](const QString& key, const QString& fallback) { const QString runtime = envValue(key); return !runtime.isEmpty() ? runtime : env.value(key, fallback); };
  deepseek_key_ = get("DEEPSEEK_API_KEY", "");
  deepseek_base_url_ = get("DEEPSEEK_BASE_URL", "https://api.deepseek.com");
  infra_url_ = get("AI_INFRA_BASE_URL", "http://127.0.0.1:8080");
  shared_secret_ = get("AI_INFRA_SHARED_SECRET", "development-only");
  workspace_root_ = QDir(get("CODE_WORKSPACE_ROOT", "..")).absolutePath();
  data_path_ = QDir::current().filePath("data/state.json");
  loadState();
}

bool BackendServer::listen(quint16 port) { return server_.listen(QHostAddress::LocalHost, port); }

void BackendServer::acceptConnection() {
  while (auto* socket = server_.nextPendingConnection()) {
    socket->setProperty("buffer", QByteArray{});
    connect(socket, &QTcpSocket::readyRead, this, [this, socket]() { readSocket(socket); });
    connect(socket, &QTcpSocket::disconnected, socket, &QObject::deleteLater);
  }
}

void BackendServer::readSocket(QTcpSocket* socket) {
  QByteArray buffer = socket->property("buffer").toByteArray() + socket->readAll();
  const int headerEnd = buffer.indexOf("\r\n\r\n"); if (headerEnd < 0) { socket->setProperty("buffer", buffer); return; }
  const QList<QByteArray> lines = buffer.left(headerEnd).split('\n'); if (lines.isEmpty()) return writeResponse(socket, {400, errorResponse("invalid request")});
  const QList<QByteArray> first = lines.first().trimmed().split(' '); if (first.size() < 2) return writeResponse(socket, {400, errorResponse("invalid request line")});
  HttpRequest request; request.method = QString::fromLatin1(first[0]); request.url = QUrl(QString::fromUtf8(first[1])); request.path = request.url.path();
  int contentLength = 0;
  for (int i = 1; i < lines.size(); ++i) { const QByteArray line = lines[i].trimmed(); const int colon = line.indexOf(':'); if (colon > 0) { const QString key = QString::fromLatin1(line.left(colon)).trimmed().toLower(); const QString value = QString::fromUtf8(line.mid(colon + 1)).trimmed(); request.headers[key] = value; if (key == "content-length") contentLength = value.toInt(); } }
  const int bodyStart = headerEnd + 4; if (buffer.size() < bodyStart + contentLength) { socket->setProperty("buffer", buffer); return; }
  if (contentLength > 0) request.json = QJsonDocument::fromJson(buffer.mid(bodyStart, contentLength)).object();
  socket->disconnect(this); dispatch(socket, request);
}

void BackendServer::dispatch(QTcpSocket* socket, HttpRequest request) {
  QPointer<QTcpSocket> guarded(socket);
  QtConcurrent::run([this, guarded, request]() { const auto response = route(request); QMetaObject::invokeMethod(this, [this, guarded, response]() { if (guarded) writeResponse(guarded, response); }, Qt::QueuedConnection); });
}

HttpResponse BackendServer::route(const HttpRequest& request) {
  if (request.path == "/api/health") { const auto health = webRequest("GET", infra_url_ + "/health"); const bool available = health.status == 200; return {200, QJsonObject{{"status", available ? "ok" : "unavailable"}, {"provider", "C++ AI Infra"}, {"model", "deepseek-chat"}, {"configured", !deepseek_key_.isEmpty()}, {"infraAvailable", available}, {"infraUrl", infra_url_}, {"backend", "C++ Qt"}}}; }
  if (request.path == "/api/providers") return {200, QJsonObject{{"providers", QJsonArray{QJsonObject{{"id", "deepseek-chat"}, {"name", "DeepSeek Chat"}, {"provider", "ai-infra"}, {"model", "deepseek-chat"}, {"description", "支持直连、C++ 基线与 AI Infra 调度"}, {"configured", !deepseek_key_.isEmpty()}, {"recommended", true}}}}, {"default", "deepseek-chat"}}};
  if (request.path == "/api/stats") { QMutexLocker lock(&state_mutex_); qint64 characters = 0; for (const auto& value : documents_) characters += static_cast<qint64>(value.toObject().value("characters").toDouble()); return {200, QJsonObject{{"documents", documents_.size()}, {"chunks", chunks_.size()}, {"conversations", conversations_.size()}, {"characters", characters}}}; }
  if (request.path == "/api/documents" && request.method == "GET") { QMutexLocker lock(&state_mutex_); return {200, QJsonObject{{"documents", documents_}}}; }
  if (request.path == "/api/documents" && request.method == "POST") {
    const QString title = request.json.value("title").toString().trimmed(), content = request.json.value("content").toString(), source = request.json.value("source").toString("manual");
    if (title.isEmpty() || content.size() < 10) return {400, errorResponse("资料名称或内容无效")};
    const QString id = QUuid::createUuid().toString(QUuid::WithoutBraces); const QStringList pieces = chunkText(content); const QString now = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
    QJsonObject document{{"id", id}, {"title", title}, {"source", source}, {"characters", content.size()}, {"chunkCount", pieces.size()}, {"createdAt", now}};
    { QMutexLocker lock(&state_mutex_); documents_.prepend(document); for (int i = 0; i < pieces.size(); ++i) chunks_.append(QJsonObject{{"id", QUuid::createUuid().toString(QUuid::WithoutBraces)}, {"documentId", id}, {"title", title}, {"source", source}, {"index", i}, {"text", pieces[i]}}); saveState(); }
    return {201, QJsonObject{{"document", document}}};
  }
  if (request.path.startsWith("/api/documents/") && request.method == "DELETE") { const QString id = request.path.section('/', -1); QMutexLocker lock(&state_mutex_); bool removed = false; for (int i = documents_.size() - 1; i >= 0; --i) if (documents_[i].toObject().value("id") == id) { documents_.removeAt(i); removed = true; } for (int i = chunks_.size() - 1; i >= 0; --i) if (chunks_[i].toObject().value("documentId") == id) chunks_.removeAt(i); saveState(); return {removed ? 200 : 404, removed ? QJsonObject{{"ok", true}} : errorResponse("文档不存在")}; }
  if (request.path == "/api/conversations") { QMutexLocker lock(&state_mutex_); QJsonArray result; for (int i = 0; i < std::min(30, conversations_.size()); ++i) result.append(conversations_[i]); return {200, QJsonObject{{"conversations", result}}}; }
  if (request.path == "/api/ask" && request.method == "POST") return ask(request.json);
  if (request.path == "/internal/infer" && request.method == "POST") return internalInfer(request);
  if (request.path == "/api/code/files" && request.method == "GET") return {200, QJsonObject{{"root", workspace_root_}, {"files", workspaceFiles()}}};
  if (request.path == "/api/code/workspace" && request.method == "PUT") { const QString root = QDir(request.json.value("root").toString()).absolutePath(); if (!QFileInfo(root).isDir()) return {400, errorResponse("工作目录不存在")}; workspace_root_ = root; return {200, QJsonObject{{"root", workspace_root_}, {"files", workspaceFiles()}}}; }
  if (request.path == "/api/code/file" && request.method == "GET") { const QString relative = QUrlQuery(request.url).queryItemValue("path", QUrl::FullyDecoded); const QString path = workspaceFile(relative); QFile file(path); if (path.isEmpty() || !file.open(QIODevice::ReadOnly)) return {400, errorResponse("文件不存在或不允许访问：" + relative)}; return {200, QJsonObject{{"path", relative}, {"content", QString::fromUtf8(file.readAll())}}}; }
  if (request.path == "/api/code/file" && request.method == "PUT") { const QString relative = request.json.value("path").toString(), path = workspaceFile(relative); QFile file(path); if (path.isEmpty() || !file.open(QIODevice::WriteOnly | QIODevice::Truncate)) return {400, errorResponse("文件无法写入")}; file.write(request.json.value("content").toString().toUtf8()); return {200, QJsonObject{{"ok", true}, {"path", relative}}}; }
  if (request.path == "/api/code/propose" && request.method == "POST") { const QString path = request.json.value("path").toString(), instruction = request.json.value("instruction").toString(), content = request.json.value("content").toString(); QString provider; const QString prompt = QString("You are editing file %1. Apply only the requested change. Return only the complete updated file, without Markdown fences.\n\nREQUEST:\n%2\n\nCURRENT FILE:\n%3").arg(path, instruction, content); try { const QString result = generate(prompt, "infra", &provider); return {200, QJsonObject{{"path", path}, {"content", stripFence(result)}, {"provider", provider}, {"model", "deepseek-chat"}}}; } catch (const std::exception& e) { return {503, errorResponse(QString::fromUtf8(e.what()))}; } }
  return {404, errorResponse("route not found")};
}

HttpResponse BackendServer::ask(const QJsonObject& body) {
  const auto started = std::chrono::steady_clock::now(); const QString question = body.value("question").toString().trimmed(); const QString modeRequested = body.value("mode").toString("chat"); const QString route = body.value("route_mode").toString(body.value("use_ai_infra").toBool(true) ? "infra" : "direct");
  if (question.size() < 2) return {400, errorResponse("问题不能为空")};
  const auto searchStarted = std::chrono::steady_clock::now(); QJsonArray contexts; QString mode = modeRequested;
  if (modeRequested == "knowledge" || (modeRequested == "auto" && !chunks_.isEmpty())) { mode = "knowledge"; contexts = retrieve(question); } else mode = "chat";
  const double searchMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - searchStarted).count(); const QString prompt = buildPrompt(question, body.value("history").toArray(), contexts, mode); QString provider;
  try {
    const auto generationStarted = std::chrono::steady_clock::now(); const QString answer = generate(prompt, route, &provider); const double generationMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - generationStarted).count(); const double totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    QJsonArray citations; for (int i = 0; i < contexts.size(); ++i) { const auto c = contexts[i].toObject(); citations.append(QJsonObject{{"id", QString("[%1]").arg(i + 1)}, {"documentId", c.value("documentId")}, {"title", c.value("title")}, {"chunk", c.value("index").toInt() + 1}, {"score", c.value("score")}, {"excerpt", c.value("text").toString().left(240)}}); }
    QJsonObject result{{"question", question}, {"answer", answer}, {"provider", provider}, {"model", "deepseek-chat"}, {"mode", mode}, {"degraded", false}, {"warning", QJsonValue()}, {"citations", citations}, {"timing", QJsonObject{{"searchMs", searchMs}, {"generationMs", generationMs}, {"totalMs", totalMs}}}};
    QJsonObject saved = result; saved["id"] = QUuid::createUuid().toString(QUuid::WithoutBraces); saved["createdAt"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs); { QMutexLocker lock(&state_mutex_); conversations_.prepend(saved); saveState(); }
    return {200, result};
  } catch (const std::exception& e) { return {503, errorResponse(QString("%1 推理失败：%2").arg(route, QString::fromUtf8(e.what())))}; }
}

HttpResponse BackendServer::internalInfer(const HttpRequest& request) { if (request.headers.value("x-aiinfra-secret") != shared_secret_) return {401, errorResponse("invalid AI Infra secret")}; try { return {200, QJsonObject{{"text", deepSeek(request.json.value("prompt").toString(), request.json.value("max_tokens").toInt(1000))}}}; } catch (const std::exception& e) { return {503, errorResponse(QString::fromUtf8(e.what()))}; } }

QString BackendServer::generate(const QString& prompt, const QString& route, QString* provider) { if (route == "infra") { *provider = "C++ AI Infra"; return aiInfra(prompt); } if (route == "baseline") { *provider = "C++ Backend 基线"; return deepSeek(prompt); } *provider = "DeepSeek Direct"; return deepSeek(prompt, 1400); }

QString BackendServer::deepSeek(const QString& prompt, int maxTokens) {
  if (deepseek_key_.isEmpty()) throw std::runtime_error("DEEPSEEK_API_KEY 未配置");
  const QJsonObject payload{{"model", "deepseek-chat"}, {"messages", QJsonArray{QJsonObject{{"role", "user"}, {"content", prompt}}}}, {"temperature", 0.3}, {"max_tokens", maxTokens}};
  const auto response = webRequest("POST", deepseek_base_url_ + "/chat/completions", QJsonDocument(payload).toJson(QJsonDocument::Compact), {{"Authorization", "Bearer " + deepseek_key_}});
  if (response.status != 200) throw std::runtime_error(QString("DeepSeek HTTP %1: %2").arg(response.status).arg(QString::fromUtf8(response.body)).toUtf8().constData());
  const QJsonArray choices = QJsonDocument::fromJson(response.body).object().value("choices").toArray();
  return choices.isEmpty() ? QString{} : choices.at(0).toObject().value("message").toObject().value("content").toString();
}

QString BackendServer::aiInfra(const QString& prompt, int maxTokens) { const QString url = infra_url_ + QString("/v1/infer?model=deepseek-chat&max_tokens=%1").arg(maxTokens); const QJsonObject payload{{"prompt", prompt}, {"max_tokens", maxTokens}}; const auto response = webRequest("POST", url, QJsonDocument(payload).toJson(QJsonDocument::Compact)); if (response.status != 200) throw std::runtime_error(QString("AI Infra HTTP %1: %2").arg(response.status).arg(QString::fromUtf8(response.body)).toUtf8().constData()); return QJsonDocument::fromJson(response.body).object().value("text").toString(); }

QString BackendServer::buildPrompt(const QString& question, const QJsonArray& history, const QJsonArray& contexts, const QString& mode) const { QString prompt = mode == "chat" ? "system: 你是知问，一个专业、友好、清晰的通用 AI 助手。使用与用户相同的语言。" : "system: 你是严谨的知识库问答助手，只能根据资料回答，并标注引用编号。"; const int historyStart = std::max(0, history.size() - (mode == "chat" ? 12 : 8)); for (int i = historyStart; i < history.size(); ++i) { const auto h = history[i].toObject(); prompt += "\n\n" + h.value("role").toString() + ": " + h.value("content").toString(); } if (mode == "knowledge") { prompt += "\n\n资料："; for (int i = 0; i < contexts.size(); ++i) { const auto c = contexts[i].toObject(); prompt += QString("\n[%1] 来源：%2\n%3").arg(i + 1).arg(c.value("title").toString(), c.value("text").toString()); } } return prompt + "\n\nuser: " + question; }

QStringList BackendServer::tokenize(const QString& text) const { QString normalized = text.toLower(); QStringList result; QRegularExpression english("[a-z0-9][a-z0-9_+.#-]*"); auto it = english.globalMatch(normalized); while (it.hasNext()) result << it.next().captured(); QRegularExpression chinese("[\\x{3400}-\\x{9fff}]+"); auto chineseIt = chinese.globalMatch(normalized); while (chineseIt.hasNext()) { const QString run = chineseIt.next().captured(); result << run; for (int i = 0; i + 1 < run.size(); ++i) result << run.mid(i, 2); } return result; }

QJsonArray BackendServer::retrieve(const QString& query, int limit) const { QMutexLocker lock(&state_mutex_); const QStringList terms = tokenize(query); if (terms.isEmpty() || chunks_.isEmpty()) return {}; struct Scored { double score; QJsonObject object; }; QVector<Scored> scored; QVector<QStringList> tokens; double avg = 0; for (const auto& value : chunks_) { tokens << tokenize(value.toObject().value("text").toString()); avg += tokens.last().size(); } avg = std::max(1.0, avg / tokens.size()); for (int i = 0; i < chunks_.size(); ++i) { double score = 0; for (const QString& term : terms) { const int tf = tokens[i].count(term); if (!tf) continue; int df = 0; for (const auto& list : tokens) if (list.contains(term)) ++df; const double idf = std::log(1.0 + (tokens.size() - df + 0.5) / (df + 0.5)); score += idf * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * tokens[i].size() / avg)); } if (score > 0.05) { auto object = chunks_[i].toObject(); object["score"] = score; scored.push_back({score, object}); } } std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) { return a.score > b.score; }); QJsonArray result; for (int i = 0; i < std::min(limit, scored.size()); ++i) result.append(scored[i].object); return result; }

QStringList BackendServer::chunkText(const QString& text) const { QString normalized = text; normalized.replace("\r\n", "\n"); normalized.replace(QRegularExpression("[\\t ]+"), " "); normalized.replace(QRegularExpression("\\n{3,}"), "\n\n"); normalized = normalized.trimmed(); QStringList result; const int size = 650, overlap = 100, step = size - overlap; for (int start = 0; start < normalized.size(); start += step) { const QString chunk = normalized.mid(start, size).trimmed(); if (chunk.size() >= 10 && !result.contains(chunk)) result << chunk; if (start + size >= normalized.size()) break; } return result; }

QString BackendServer::workspaceFile(const QString& relative) const { static const QSet<QString> allowed{"py","js","jsx","ts","tsx","css","html","json","yaml","yml","toml","ini","txt","sql","c","cc","cpp","h","hpp","pro"}; const QString root = QDir::fromNativeSeparators(QFileInfo(workspace_root_).canonicalFilePath()); const QString candidate = QDir::fromNativeSeparators(QFileInfo(QDir(root).filePath(relative)).canonicalFilePath()); if (root.isEmpty() || candidate.isEmpty() || (!candidate.startsWith(root + "/") && candidate != root) || !allowed.contains(QFileInfo(candidate).suffix().toLower())) return {}; return candidate; }

QJsonArray BackendServer::workspaceFiles() const { static const QSet<QString> ignored{".git","node_modules","dist","build","build-windows",".venv","__pycache__",".pytest_cache"}; QJsonArray result; QDir root(workspace_root_); QDirIterator it(workspace_root_, QDir::Files, QDirIterator::Subdirectories); while (it.hasNext() && result.size() < 800) { const QString path = it.next(), relative = QDir::fromNativeSeparators(root.relativeFilePath(path)); bool skip = false; for (const QString& part : relative.split('/')) if (ignored.contains(part)) { skip = true; break; } if (!skip && !workspaceFile(relative).isEmpty()) result.append(relative); } return result; }

void BackendServer::loadState() { QDir().mkpath(QFileInfo(data_path_).absolutePath()); QFile file(data_path_); if (!file.open(QIODevice::ReadOnly)) return; const auto root = QJsonDocument::fromJson(file.readAll()).object(); documents_ = root.value("documents").toArray(); chunks_ = root.value("chunks").toArray(); conversations_ = root.value("conversations").toArray(); }

void BackendServer::saveState() { QFile file(data_path_); if (file.open(QIODevice::WriteOnly | QIODevice::Truncate)) file.write(QJsonDocument(QJsonObject{{"documents", documents_}, {"chunks", chunks_}, {"conversations", conversations_}}).toJson(QJsonDocument::Compact)); }

void BackendServer::writeResponse(QTcpSocket* socket, const HttpResponse& response) { const QByteArray body = QJsonDocument(response.json).toJson(QJsonDocument::Compact); QByteArray header = QString("HTTP/1.1 %1 %2\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\nContent-Length: %3\r\n\r\n").arg(response.status).arg(statusText(response.status)).arg(body.size()).toUtf8(); socket->write(header + body); socket->disconnectFromHost(); }

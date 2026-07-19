#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QMutex>
#include <QObject>
#include <QTcpServer>
#include <QUrl>

class QTcpSocket;

struct HttpRequest {
  QString method;
  QString path;
  QUrl url;
  QJsonObject json;
  QMap<QString, QString> headers;
};

struct HttpResponse {
  int status{200};
  QJsonObject json;
};

class BackendServer : public QObject {
  Q_OBJECT
 public:
  explicit BackendServer(QObject* parent = nullptr);
  bool listen(quint16 port = 8000);

 private:
  void acceptConnection();
  void readSocket(QTcpSocket* socket);
  void dispatch(QTcpSocket* socket, HttpRequest request);
  HttpResponse route(const HttpRequest& request);
  HttpResponse ask(const QJsonObject& body);
  HttpResponse internalInfer(const HttpRequest& request);
  QString generate(const QString& prompt, const QString& route, QString* provider);
  QString deepSeek(const QString& prompt, int maxTokens = 1000);
  QString aiInfra(const QString& prompt, int maxTokens = 1000);
  QString buildPrompt(const QString& question, const QJsonArray& history, const QJsonArray& contexts, const QString& mode) const;
  QJsonArray retrieve(const QString& query, int limit = 6) const;
  QStringList tokenize(const QString& text) const;
  QStringList chunkText(const QString& text) const;
  QString workspaceFile(const QString& relative) const;
  QJsonArray workspaceFiles() const;
  void loadState();
  void saveState();
  void writeResponse(QTcpSocket* socket, const HttpResponse& response);

  QTcpServer server_;
  mutable QMutex state_mutex_;
  QJsonArray documents_;
  QJsonArray chunks_;
  QJsonArray conversations_;
  QString workspace_root_;
  QString data_path_;
  QString deepseek_key_;
  QString deepseek_base_url_;
  QString infra_url_;
  QString shared_secret_;
};

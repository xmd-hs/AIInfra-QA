#include "mainwindow.h"

#include <QButtonGroup>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QScrollBar>
#include <QSplitter>
#include <QStackedWidget>
#include <QStyle>
#include <QTextBrowser>
#include <QTextEdit>
#include <QTimer>
#include <QTreeWidget>
#include <QUrlQuery>
#include <QVBoxLayout>

namespace {
QLabel* label(const QString& text, const char* role = nullptr) {
  auto* item = new QLabel(text);
  if (role) item->setProperty("role", role);
  item->setWordWrap(true);
  return item;
}

QFrame* card() {
  auto* frame = new QFrame;
  frame->setProperty("card", true);
  return frame;
}

QString htmlEscape(QString value) {
  return value.toHtmlEscaped().replace("\n", "<br>");
}
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), api_(this) {
  resize(1080, 760);
  setMinimumSize(960, 650);
  setWindowTitle("知问 · AI Infra Knowledge Copilot");

  auto* root = new QWidget;
  auto* layout = new QHBoxLayout(root);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(0);
  layout->addWidget(buildSidebar());
  pages_ = new QStackedWidget;
  pages_->addWidget(buildChatPage());
  pages_->addWidget(buildKnowledgePage());
  pages_->addWidget(buildCodePage());
  pages_->addWidget(buildHistoryPage());
  pages_->addWidget(buildModelPage());
  layout->addWidget(pages_, 1);
  setCentralWidget(root);

  toast_ = new QLabel(this);
  toast_->setProperty("toast", true);
  toast_->hide();

  setStyleSheet(R"(
    QMainWindow,QStackedWidget{background:#f3f5f1;color:#172822}
    QWidget#sidebar{background:#14342b;color:white;min-width:236px;max-width:236px}
    QLabel[role="brand"]{font-size:21px;font-weight:800;color:white}
    QLabel[role="brandIcon"]{background:#d9f264;color:#173028;border-radius:13px;font-size:20px;font-weight:900;padding:11px}
    QLabel[role="muted"]{color:#87958f;font-size:11px} QLabel[role="eyebrow"]{color:#81952c;font-size:10px;font-weight:800}
    QLabel[role="title"]{color:#172822;font-size:30px;font-weight:800} QLabel[role="stat"]{font-size:20px;font-weight:800}
    QPushButton[nav="true"]{border:0;text-align:left;color:#a6bbb3;padding:12px 14px;border-radius:10px;background:transparent}
    QPushButton[nav="true"]:hover,QPushButton[nav="true"][active="true"]{background:#23493d;color:white;font-weight:700}
    QFrame[card="true"]{background:white;border:1px solid #dce3de;border-radius:16px}
    QPushButton{border:1px solid #d8dfda;border-radius:9px;padding:8px 13px;background:#fafbf9;color:#31483f}
    QPushButton:hover{border-color:#94a83b} QPushButton:disabled{color:#a9b2ae;background:#f0f2ef}
    QPushButton[primary="true"]{border:0;background:#d9f264;color:#18372e;font-weight:800;padding:10px 18px}
    QPushButton[route="true"]{border:0;background:#edf1ed;color:#687770;padding:8px 12px}
    QPushButton[route="true"]:checked{background:#18372e;color:#d9f264;font-weight:800}
    QLineEdit,QTextEdit,QComboBox,QListWidget,QTextBrowser{border:1px solid #d9e1dc;border-radius:9px;background:white;padding:8px;selection-background-color:#d9f264;selection-color:#18372e}
    QLineEdit:focus,QTextEdit:focus,QComboBox:focus{border-color:#91a43b}
    QTextBrowser#chat{background:white;border:0;padding:18px;font-size:13px}
    QListWidget::item{padding:8px;border-radius:7px} QListWidget::item:selected{background:#eaf1d3;color:#273b34}
    QTextEdit[code="true"]{background:#142b25;color:#e0ebe5;font-family:Consolas;font-size:12px;border:0}
    QLabel[toast="true"]{background:#18382f;color:white;border-radius:10px;padding:11px 16px}
  )");

  selectPage(0);
  refreshOverview();
  loadDocuments();
  loadHistory();
  loadCodeFiles();
}

QWidget* MainWindow::buildSidebar() {
  auto* side = new QWidget;
  side->setObjectName("sidebar");
  auto* box = new QVBoxLayout(side);
  box->setContentsMargins(18, 25, 18, 20);
  box->setSpacing(8);

  auto* brand = new QHBoxLayout;
  brand->addWidget(label("知", "brandIcon"));
  auto* names = new QVBoxLayout;
  names->addWidget(label("知问", "brand"));
  names->addWidget(label("KNOWLEDGE COPILOT", "muted"));
  brand->addLayout(names, 1);
  box->addLayout(brand);
  box->addSpacing(24);

  box->addWidget(navButton("✦", "智能问答", 0));
  box->addWidget(navButton("▦", "知识空间", 1));
  box->addWidget(navButton("</>", "代码工作台", 2));
  box->addWidget(navButton("◷", "问答记录", 3));
  box->addWidget(navButton("◉", "模型中心", 4));
  box->addStretch();

  auto* status = card();
  auto* status_box = new QVBoxLayout(status);
  status_box->addWidget(label("AI INFRA STATUS", "muted"));
  infra_state_ = label("正在连接……");
  infra_state_->setStyleSheet("font-weight:700;color:white");
  status_box->addWidget(infra_state_);
  status->setStyleSheet("QFrame{background:#1c4035;border:1px solid #315449}");
  box->addWidget(status);
  return side;
}

QPushButton* MainWindow::navButton(const QString& icon, const QString& text, int page) {
  auto* button = new QPushButton(icon + "   " + text);
  button->setProperty("nav", true);
  button->setCursor(Qt::PointingHandCursor);
  connect(button, &QPushButton::clicked, this, [this, page]() { selectPage(page); });
  nav_buttons_.append(button);
  return button;
}

void MainWindow::selectPage(int page) {
  if (pages_) pages_->setCurrentIndex(page);
  for (int i = 0; i < nav_buttons_.size(); ++i) {
    nav_buttons_[i]->setProperty("active", i == page);
    nav_buttons_[i]->style()->unpolish(nav_buttons_[i]);
    nav_buttons_[i]->style()->polish(nav_buttons_[i]);
  }
  if (page == 1) loadDocuments();
  if (page == 2) loadCodeFiles();
  if (page == 3) loadHistory();
  if (page == 4) refreshOverview();
}

QWidget* MainWindow::pageHeader(const QString& eyebrow, const QString& title, const QString& description) {
  auto* widget = new QWidget;
  auto* box = new QVBoxLayout(widget);
  box->setContentsMargins(0, 0, 0, 10);
  box->addWidget(label(eyebrow, "eyebrow"));
  box->addWidget(label(title, "title"));
  box->addWidget(label(description, "muted"));
  return widget;
}

QWidget* MainWindow::buildChatPage() {
  auto* page = new QWidget;
  auto* box = new QVBoxLayout(page);
  box->setContentsMargins(48, 34, 48, 34);
  box->addWidget(pageHeader("AI KNOWLEDGE COPILOT", "与你的 AI 一起思考", "自由对话，也可以切换到知识库模式获得带引用的回答。"));

  auto* stats_row = new QHBoxLayout;
  const QString names[] = {"知识文档", "可检索片段", "历史问答", "知识字符"};
  for (int i = 0; i < 4; ++i) {
    auto* item = card(); auto* item_box = new QVBoxLayout(item);
    stats_[i] = label("0", "stat"); item_box->addWidget(stats_[i]); item_box->addWidget(label(names[i], "muted"));
    stats_row->addWidget(item);
  }
  box->addLayout(stats_row);

  auto* shell = card();
  auto* shell_box = new QVBoxLayout(shell); shell_box->setContentsMargins(0, 0, 0, 0);
  auto* toolbar = new QWidget; auto* tools = new QHBoxLayout(toolbar);
  chat_mode_ = new QComboBox; chat_mode_->addItem("✦ 通用对话", "chat"); chat_mode_->addItem("▦ 知识库问答", "knowledge"); chat_mode_->addItem("◈ 自动模式", "auto");
  tools->addWidget(chat_mode_); tools->addStretch();
  route_group_ = new QButtonGroup(this); route_group_->setExclusive(true);
  const QStringList routes = {"DeepSeek 直连", "C++ 基线", "AI Infra 加速"};
  const QStringList ids = {"direct", "baseline", "infra"};
  for (int i = 0; i < routes.size(); ++i) { auto* b = new QPushButton(routes[i]); b->setCheckable(true); b->setProperty("route", true); b->setProperty("routeId", ids[i]); route_group_->addButton(b); tools->addWidget(b); if (i == 2) b->setChecked(true); }
  shell_box->addWidget(toolbar);
  chat_messages_ = new QTextBrowser; chat_messages_->setObjectName("chat"); shell_box->addWidget(chat_messages_, 1);
  appendMessage("assistant", "你好，我是知问。你可以选择 DeepSeek 直连、C++ 基线或 AI Infra 加速进行问答。", "C++ AI Infra · DeepSeek Chat");
  auto* compose = new QWidget; auto* compose_box = new QHBoxLayout(compose);
  chat_input_ = new QTextEdit; chat_input_->setPlaceholderText("向当前模型询问任何问题……"); chat_input_->setMaximumHeight(88);
  send_button_ = new QPushButton("发送 ↑"); send_button_->setProperty("primary", true); send_button_->setMinimumHeight(56);
  connect(send_button_, &QPushButton::clicked, this, &MainWindow::sendQuestion);
  compose_box->addWidget(chat_input_, 1); compose_box->addWidget(send_button_); shell_box->addWidget(compose);
  box->addWidget(shell, 1);
  return page;
}

void MainWindow::appendMessage(const QString& role, const QString& text, const QString& meta) {
  if (!chat_messages_) return;
  const bool user = role == "user";
  const QString background = user ? "#1b3a31" : "#f4f7f4";
  const QString color = user ? "white" : "#24362f";
  const QString align = user ? "right" : "left";
  chat_messages_->append(QString("<div align='%1'><table width='75%%' cellpadding='12' style='background:%2;color:%3;border-radius:12px'><tr><td>%4%5</td></tr></table></div><br>")
      .arg(align, background, color, htmlEscape(text), meta.isEmpty() ? "" : "<br><small style='color:#82908a'>" + htmlEscape(meta) + "</small>"));
  chat_messages_->verticalScrollBar()->setValue(chat_messages_->verticalScrollBar()->maximum());
}

void MainWindow::sendQuestion() {
  const QString question = chat_input_->toPlainText().trimmed();
  if (question.isEmpty()) return;
  appendMessage("user", question);
  chat_input_->clear(); send_button_->setEnabled(false); send_button_->setText("思考中…");
  QString route = "infra";
  if (auto* checked = route_group_->checkedButton()) route = checked->property("routeId").toString();
  QJsonObject body{{"question", question}, {"model_id", "deepseek-chat"}, {"mode", chat_mode_->currentData().toString()}, {"route_mode", route}, {"use_ai_infra", route == "infra"}, {"history", history_context_}};
  api_.send("POST", "/api/ask", body, [this, question](const QJsonDocument& doc, const QString& error) {
    send_button_->setEnabled(true); send_button_->setText("发送 ↑");
    if (!error.isEmpty()) { appendMessage("assistant", error, "请求失败"); return; }
    const auto object = doc.object(); const auto timing = object.value("timing").toObject();
    appendMessage("assistant", object.value("answer").toString(), QString("%1 · %2 · %3ms").arg(object.value("provider").toString(), object.value("model").toString()).arg(timing.value("totalMs").toDouble()));
    history_context_.append(QJsonObject{{"role", "user"}, {"content", question}});
    history_context_.append(QJsonObject{{"role", "assistant"}, {"content", object.value("answer").toString()}});
    while (history_context_.size() > 12) history_context_.removeFirst();
    refreshOverview();
  });
}

QWidget* MainWindow::buildKnowledgePage() {
  auto* page = new QWidget; auto* box = new QVBoxLayout(page); box->setContentsMargins(48, 34, 48, 34);
  box->addWidget(pageHeader("KNOWLEDGE WORKSPACE", "构建你的知识空间", "添加文本资料，系统将自动分块并建立 BM25 检索索引。"));
  auto* split = new QSplitter;
  auto* form = card(); auto* form_box = new QVBoxLayout(form); form_box->addWidget(label("添加知识", "title"));
  knowledge_title_ = new QLineEdit; knowledge_title_->setPlaceholderText("资料名称"); knowledge_content_ = new QTextEdit; knowledge_content_->setPlaceholderText("粘贴 TXT、Markdown、JSON 或 CSV 内容……");
  auto* add = new QPushButton("加入知识空间"); add->setProperty("primary", true);
  connect(add, &QPushButton::clicked, this, [this]() { QJsonObject body{{"title", knowledge_title_->text().trimmed()}, {"content", knowledge_content_->toPlainText()}, {"source", "qt-client"}}; api_.send("POST", "/api/documents", body, [this](const QJsonDocument&, const QString& error) { if (!error.isEmpty()) return showToast(error, true); knowledge_title_->clear(); knowledge_content_->clear(); showToast("资料已加入知识空间"); loadDocuments(); }); });
  form_box->addWidget(knowledge_title_); form_box->addWidget(knowledge_content_, 1); form_box->addWidget(add);
  auto* list = card(); auto* list_box = new QVBoxLayout(list); list_box->addWidget(label("知识文档", "title")); document_list_ = new QListWidget; list_box->addWidget(document_list_);
  split->addWidget(form); split->addWidget(list); split->setSizes({550, 550}); box->addWidget(split, 1); return page;
}

void MainWindow::loadDocuments() { if (!document_list_) return; api_.get("/api/documents", [this](const QJsonDocument& doc, const QString& error) { if (!error.isEmpty()) return; document_list_->clear(); for (const auto& value : doc.object().value("documents").toArray()) { const auto o = value.toObject(); document_list_->addItem(o.value("title").toString() + "\n" + o.value("source").toString()); } refreshOverview(); }); }

QWidget* MainWindow::buildCodePage() {
  auto* page = new QWidget; auto* box = new QVBoxLayout(page); box->setContentsMargins(24, 24, 24, 24);
  box->addWidget(pageHeader("AGENTIC DEVELOPMENT ENVIRONMENT", "AI Agent 工作台", "项目浏览、代码编辑、任务规划、Agent 对话与 Diff 审核集中在同一个工作区。"));
  auto* picker = new QHBoxLayout; workspace_path_ = new QLineEdit; workspace_path_->setPlaceholderText("本地项目完整路径"); auto* browse = new QPushButton("浏览…"); auto* switcher = new QPushButton("切换目录");
  connect(browse, &QPushButton::clicked, this, [this]() { const auto dir = QFileDialog::getExistingDirectory(this, "选择代码工作目录", workspace_path_->text()); if (!dir.isEmpty()) workspace_path_->setText(QDir::toNativeSeparators(dir)); });
  connect(switcher, &QPushButton::clicked, this, [this]() { api_.send("PUT", "/api/code/workspace", QJsonObject{{"root", workspace_path_->text()}}, [this](const QJsonDocument&, const QString& error) { if (!error.isEmpty()) return showToast(error, true); showToast("工作目录已切换"); loadCodeFiles(); }); });
  picker->addWidget(workspace_path_, 1); picker->addWidget(browse); picker->addWidget(switcher); box->addLayout(picker);

  auto* split = new QSplitter;
  auto* explorer = card(); auto* explorer_box = new QVBoxLayout(explorer); explorer_box->setContentsMargins(8, 12, 8, 8);
  explorer_box->addWidget(label("EXPLORER", "eyebrow"));
  code_tree_ = new QTreeWidget; code_tree_->setHeaderHidden(true); code_tree_->setMinimumWidth(170);
  connect(code_tree_, &QTreeWidget::itemClicked, this, [this](QTreeWidgetItem* item) { openCodeFile(item); });
  explorer_box->addWidget(code_tree_);

  auto* editor_card = card(); auto* editor_box = new QVBoxLayout(editor_card); editor_box->setContentsMargins(0, 0, 0, 0);
  auto* title_bar = new QWidget; title_bar->setStyleSheet("background:#eef2ef;border-bottom:1px solid #d9e0dc"); auto* title_row = new QHBoxLayout(title_bar); title_row->setContentsMargins(13, 7, 9, 7);
  code_file_name_ = label("请选择代码文件"); code_file_name_->setStyleSheet("font-family:Consolas;font-weight:700"); auto* save = new QPushButton("保存");
  connect(save, &QPushButton::clicked, this, [this]() { if (code_file_name_->text().isEmpty()) return; api_.send("PUT", "/api/code/file", QJsonObject{{"path", code_file_name_->text()}, {"content", code_editor_->toPlainText()}}, [this](const QJsonDocument&, const QString& error) { showToast(error.isEmpty() ? "代码已保存" : error, !error.isEmpty()); }); });
  title_row->addWidget(code_file_name_, 1); title_row->addWidget(save); editor_box->addWidget(title_bar);
  code_editor_ = new QTextEdit; code_editor_->setProperty("code", true); code_editor_->setLineWrapMode(QTextEdit::NoWrap); editor_box->addWidget(code_editor_, 1);

  auto* agent = card(); agent->setMinimumWidth(260); auto* agent_box = new QVBoxLayout(agent); agent_box->setContentsMargins(12, 12, 12, 12);
  auto* agent_title = new QHBoxLayout; agent_title->addWidget(label("AGENT", "eyebrow")); agent_title->addStretch(); agent_title->addWidget(label("DeepSeek · AI Infra", "muted")); agent_box->addLayout(agent_title);
  agent_chat_ = new QTextBrowser; agent_chat_->setOpenExternalLinks(false); agent_chat_->setHtml("<b>Agent</b><br><span style='color:#718079'>选择文件并描述任务，我会分析代码、规划步骤并生成可审核的修改。</span>"); agent_chat_->setMaximumHeight(150); agent_box->addWidget(agent_chat_);
  agent_box->addWidget(label("任务步骤", "muted")); agent_steps_ = new QListWidget; agent_steps_->setMaximumHeight(125); agent_steps_->addItem("○ 等待任务"); agent_box->addWidget(agent_steps_);
  code_instruction_ = new QTextEdit; code_instruction_->setMaximumHeight(90); code_instruction_->setPlaceholderText("向 Agent 描述任务…\n例如：重构网络请求并补充错误处理"); agent_box->addWidget(code_instruction_);
  auto* propose = new QPushButton("运行 Agent  ↵"); propose->setProperty("primary", true); agent_box->addWidget(propose);
  agent_box->addWidget(label("变更 Diff", "muted")); diff_view_ = new QTextBrowser; diff_view_->setStyleSheet("font-family:Consolas;background:#142b25;color:#dfe9e3;border:0"); diff_view_->setPlaceholderText("Agent 生成的代码差异将在这里显示"); agent_box->addWidget(diff_view_, 1);
  code_proposal_ = new QTextEdit; code_proposal_->hide();
  auto* actions = new QHBoxLayout; auto* reject = new QPushButton("拒绝"); auto* apply = new QPushButton("应用修改"); apply->setProperty("primary", true); actions->addWidget(reject); actions->addWidget(apply); agent_box->addLayout(actions);
  connect(reject, &QPushButton::clicked, this, [this]() { code_proposal_->clear(); diff_view_->clear(); agent_steps_->clear(); agent_steps_->addItem("○ 修改已拒绝"); showToast("已拒绝 Agent 修改"); });
  connect(apply, &QPushButton::clicked, this, [this]() { if (code_proposal_->toPlainText().isEmpty()) return; code_editor_->setPlainText(code_proposal_->toPlainText()); agent_steps_->addItem("✓ 修改已应用到编辑器"); showToast("修改已应用，请保存后写入磁盘"); });
  connect(propose, &QPushButton::clicked, this, [this, propose]() {
    if (code_file_name_->text().isEmpty() || code_instruction_->toPlainText().trimmed().isEmpty()) return showToast("请先选择文件并输入任务", true);
    propose->setEnabled(false); propose->setText("Agent 正在执行…"); agent_steps_->clear(); agent_steps_->addItem("✓ 读取当前文件"); agent_steps_->addItem("● 分析任务与代码上下文"); agent_steps_->addItem("○ 生成修改方案"); agent_steps_->addItem("○ 构建 Diff");
    agent_chat_->append("<br><b>你</b><br>" + htmlEscape(code_instruction_->toPlainText()));
    QJsonObject body{{"path", code_file_name_->text()}, {"instruction", code_instruction_->toPlainText()}, {"content", code_editor_->toPlainText()}};
    api_.send("POST", "/api/code/propose", body, [this, propose](const QJsonDocument& doc, const QString& error) {
      propose->setEnabled(true); propose->setText("运行 Agent  ↵"); if (!error.isEmpty()) { agent_steps_->addItem("✕ Agent 执行失败"); return showToast(error, true); }
      const QString before = code_editor_->toPlainText(), after = doc.object().value("content").toString(); code_proposal_->setPlainText(after);
      agent_steps_->clear(); agent_steps_->addItem("✓ 读取当前文件"); agent_steps_->addItem("✓ 分析任务与代码上下文"); agent_steps_->addItem("✓ 生成修改方案"); agent_steps_->addItem("✓ Diff 等待审核");
      const QStringList oldLines = before.split('\n'), newLines = after.split('\n'); QString diff = "<pre>"; const int count = qMax(oldLines.size(), newLines.size());
      for (int i = 0; i < count; ++i) { const QString oldLine = i < oldLines.size() ? oldLines[i] : QString(); const QString newLine = i < newLines.size() ? newLines[i] : QString(); if (oldLine == newLine) { if (!oldLine.trimmed().isEmpty()) diff += "  " + htmlEscape(oldLine) + "\n"; } else { if (i < oldLines.size()) diff += "<span style='color:#ff9b9b'>- " + htmlEscape(oldLine) + "</span>\n"; if (i < newLines.size()) diff += "<span style='color:#b8ef8b'>+ " + htmlEscape(newLine) + "</span>\n"; } } diff += "</pre>"; diff_view_->setHtml(diff);
      agent_chat_->append("<br><b>Agent</b><br>修改方案已生成。请检查右侧 Diff，然后选择应用或拒绝。"); showToast("Agent 已完成，等待审核");
    });
  });
  split->addWidget(explorer); split->addWidget(editor_card); split->addWidget(agent); split->setSizes({180, 420, 280}); split->setStretchFactor(1, 1); box->addWidget(split, 1); return page;
}

void MainWindow::loadCodeFiles() { if (!code_tree_) return; api_.get("/api/code/files", [this](const QJsonDocument& doc, const QString& error) { if (!error.isEmpty()) return; const auto object = doc.object(); workspace_path_->setText(object.value("root").toString()); code_tree_->clear(); QMap<QString, QTreeWidgetItem*> folders; for (const auto& value : object.value("files").toArray()) { const QString path = value.toString(); const QStringList parts = path.split('/'); QTreeWidgetItem* parent = nullptr; QString accumulated; for (int i = 0; i < parts.size(); ++i) { accumulated += (accumulated.isEmpty() ? "" : "/") + parts[i]; if (i == parts.size() - 1) { auto* file = parent ? new QTreeWidgetItem(parent) : new QTreeWidgetItem(code_tree_); file->setText(0, parts[i]); file->setData(0, Qt::UserRole, path); } else { if (!folders.contains(accumulated)) { auto* folder = parent ? new QTreeWidgetItem(parent) : new QTreeWidgetItem(code_tree_); folder->setText(0, "▸ " + parts[i]); folders[accumulated] = folder; } parent = folders[accumulated]; } } } code_tree_->expandToDepth(0); }); }

void MainWindow::openCodeFile(QTreeWidgetItem* item) { if (!item) item = code_tree_->currentItem(); if (!item) return; const QString path = item->data(0, Qt::UserRole).toString(); if (path.isEmpty()) { item->setExpanded(!item->isExpanded()); return; } api_.get("/api/code/file?path=" + QString::fromUtf8(QUrl::toPercentEncoding(path)), [this, path](const QJsonDocument& doc, const QString& error) { if (!error.isEmpty()) return showToast(error, true); code_file_name_->setText(path); code_editor_->setPlainText(doc.object().value("content").toString()); code_proposal_->clear(); diff_view_->clear(); agent_steps_->clear(); agent_steps_->addItem("○ 等待任务"); }); }

QWidget* MainWindow::buildHistoryPage() { auto* page = new QWidget; auto* box = new QVBoxLayout(page); box->setContentsMargins(48, 34, 48, 34); box->addWidget(pageHeader("CONVERSATION ARCHIVE", "问答记录", "查看问题、模型、响应时间和回答内容。")); history_list_ = new QListWidget; box->addWidget(history_list_, 1); return page; }

void MainWindow::loadHistory() { if (!history_list_) return; api_.get("/api/conversations", [this](const QJsonDocument& doc, const QString& error) { if (!error.isEmpty()) return; history_list_->clear(); for (const auto& value : doc.object().value("conversations").toArray()) { const auto o = value.toObject(); const auto timing = o.value("timing").toObject(); history_list_->addItem(QString("%1\n%2 · %3 · %4ms\n%5").arg(o.value("question").toString(), o.value("provider").toString(), o.value("model").toString()).arg(timing.value("totalMs").toDouble()).arg(o.value("answer").toString())); } }); }

QWidget* MainWindow::buildModelPage() { auto* page = new QWidget; auto* box = new QVBoxLayout(page); box->setContentsMargins(48, 34, 48, 34); box->addWidget(pageHeader("MODEL ORCHESTRATION", "模型与基础设施", "查看 DeepSeek 配置和 C++ AI Infra 运行状态。")); auto* status = card(); auto* status_box = new QVBoxLayout(status); status_box->addWidget(label("C++ AI Infra", "title")); model_health_ = label("正在检查服务状态……", "muted"); status_box->addWidget(model_health_); auto* flow = label("Qt 5.14 客户端  →  C++ Backend RAG  →  C++ AI Infra  →  DeepSeek", "stat"); flow->setAlignment(Qt::AlignCenter); status_box->addSpacing(30); status_box->addWidget(flow); status_box->addStretch(); box->addWidget(status, 1); return page; }

void MainWindow::refreshOverview() {
  api_.get("/api/health", [this](const QJsonDocument& doc, const QString& error) { const auto o = doc.object(); const bool ok = error.isEmpty() && o.value("infraAvailable").toBool(); infra_state_->setText(ok ? "● AI Infra 已连接" : "○ AI Infra 未连接"); if (model_health_) model_health_->setText(ok ? QString("运行正常 · %1 · %2").arg(o.value("model").toString(), o.value("infraUrl").toString()) : "服务未连接：" + error); });
  api_.get("/api/stats", [this](const QJsonDocument& doc, const QString& error) { if (!error.isEmpty()) return; const auto o = doc.object(); const QString keys[] = {"documents", "chunks", "conversations", "characters"}; for (int i = 0; i < 4; ++i) if (stats_[i]) stats_[i]->setText(QString::number(o.value(keys[i]).toDouble(), 'f', 0)); });
}

void MainWindow::showToast(const QString& message, bool error) {
  toast_->setText(message); toast_->setStyleSheet(error ? "background:#8a3939;color:white;padding:11px 16px;border-radius:10px" : "background:#18382f;color:white;padding:11px 16px;border-radius:10px");
  toast_->adjustSize(); toast_->move(width() - toast_->width() - 28, height() - toast_->height() - 28); toast_->show(); toast_->raise(); QTimer::singleShot(3200, toast_, &QLabel::hide);
}

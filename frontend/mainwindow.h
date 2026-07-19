#pragma once

#include "api_client.h"

#include <QJsonArray>
#include <QMainWindow>

class QButtonGroup;
class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QStackedWidget;
class QTextBrowser;
class QTextEdit;

class MainWindow : public QMainWindow {
  Q_OBJECT
 public:
  explicit MainWindow(QWidget* parent = nullptr);

 private:
  QWidget* buildSidebar();
  QWidget* buildChatPage();
  QWidget* buildKnowledgePage();
  QWidget* buildCodePage();
  QWidget* buildHistoryPage();
  QWidget* buildModelPage();
  QWidget* pageHeader(const QString& eyebrow, const QString& title, const QString& description);
  QPushButton* navButton(const QString& icon, const QString& text, int page);
  void selectPage(int page);
  void refreshOverview();
  void sendQuestion();
  void appendMessage(const QString& role, const QString& text, const QString& meta = {});
  void loadDocuments();
  void loadHistory();
  void loadCodeFiles();
  void openCodeFile();
  void showToast(const QString& message, bool error = false);

  ApiClient api_;
  QStackedWidget* pages_{};
  QList<QPushButton*> nav_buttons_;
  QLabel* infra_state_{};
  QLabel* stats_[4]{};

  QTextBrowser* chat_messages_{};
  QTextEdit* chat_input_{};
  QComboBox* chat_mode_{};
  QButtonGroup* route_group_{};
  QPushButton* send_button_{};
  QJsonArray history_context_;

  QLineEdit* knowledge_title_{};
  QTextEdit* knowledge_content_{};
  QListWidget* document_list_{};

  QLineEdit* workspace_path_{};
  QListWidget* code_files_{};
  QTextEdit* code_editor_{};
  QTextEdit* code_instruction_{};
  QTextEdit* code_proposal_{};
  QLabel* code_file_name_{};

  QListWidget* history_list_{};
  QLabel* model_health_{};
  QLabel* toast_{};
};

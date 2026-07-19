#include "mainwindow.h"

#include <QApplication>
#include <QFont>

int main(int argc, char* argv[]) {
  QApplication app(argc, argv);
  app.setApplicationName("知问 AI Infra");
  app.setFont(QFont("Microsoft YaHei", 9));
  MainWindow window;
  window.show();
  return app.exec();
}

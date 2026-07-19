#include "server.h"

#include <QCoreApplication>
#include <QTextStream>

int main(int argc, char* argv[]) {
  QCoreApplication app(argc, argv);
  BackendServer server;
  if (!server.listen(8000)) {
    QTextStream(stderr) << "Failed to listen on port 8000\n";
    return 1;
  }
  QTextStream(stdout) << "C++ backend listening on http://127.0.0.1:8000\n";
  return app.exec();
}

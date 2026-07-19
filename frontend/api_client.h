#pragma once

#include <QJsonDocument>
#include <QNetworkAccessManager>
#include <QObject>
#include <functional>

class QNetworkReply;

class ApiClient : public QObject {
  Q_OBJECT
 public:
  using Callback = std::function<void(const QJsonDocument&, const QString&)>;
  explicit ApiClient(QObject* parent = nullptr);
  void get(const QString& path, Callback callback);
  void send(const QString& method, const QString& path, const QJsonObject& body, Callback callback);

 private:
  void finish(QNetworkReply* reply, Callback callback);
  QNetworkAccessManager manager_;
  QString base_url_{"http://127.0.0.1:8000"};
};

#include "api_client.h"

#include <QJsonObject>
#include <QNetworkReply>
#include <QNetworkRequest>

ApiClient::ApiClient(QObject* parent) : QObject(parent) {}

void ApiClient::get(const QString& path, Callback callback) {
  auto* reply = manager_.get(QNetworkRequest(QUrl(base_url_ + path)));
  connect(reply, &QNetworkReply::finished, this, [this, reply, callback]() { finish(reply, callback); });
}

void ApiClient::send(const QString& method, const QString& path, const QJsonObject& body, Callback callback) {
  QNetworkRequest request(QUrl(base_url_ + path));
  request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
  const QByteArray payload = QJsonDocument(body).toJson(QJsonDocument::Compact);
  QNetworkReply* reply = method == "PUT" ? manager_.put(request, payload) : manager_.post(request, payload);
  connect(reply, &QNetworkReply::finished, this, [this, reply, callback]() { finish(reply, callback); });
}

void ApiClient::finish(QNetworkReply* reply, Callback callback) {
  const QByteArray payload = reply->readAll();
  QString error;
  QJsonParseError parse_error;
  const auto document = QJsonDocument::fromJson(payload, &parse_error);
  if (reply->error() != QNetworkReply::NoError) {
    error = document.object().value("detail").toString(reply->errorString());
  } else if (parse_error.error != QJsonParseError::NoError) {
    error = parse_error.errorString();
  }
  reply->deleteLater();
  callback(document, error);
}

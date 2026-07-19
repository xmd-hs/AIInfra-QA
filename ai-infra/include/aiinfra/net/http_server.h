#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace aiinfra::net {

struct HttpRequest {
  std::string method;
  std::string path;
  std::unordered_map<std::string, std::string> query;
  std::unordered_map<std::string, std::string> headers;
  std::string body;
};

struct HttpResponse {
  int status{200};
  std::string content_type{"application/json"};
  std::string body;
};

using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;

class HttpServer {
 public:
  explicit HttpServer(std::uint16_t port);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  void route(const std::string& path, HttpHandler handler);
  void start();
  void stop();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::string query_value(const HttpRequest& request, const std::string& key, const std::string& fallback);
std::string json_escape(const std::string& input);

}  // namespace aiinfra::net

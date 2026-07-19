#include "aiinfra/net/http_server.h"

#include <asio.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace aiinfra::net {
namespace {

constexpr std::size_t kMaxRequestBytes = 1024 * 1024;

int hex_value(char ch) {
  if (ch >= '0' && ch <= '9') return ch - '0';
  if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
  if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
  return -1;
}

std::string url_decode(const std::string& value) {
  std::string decoded;
  decoded.reserve(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '+') {
      decoded.push_back(' ');
    } else if (value[i] == '%' && i + 2 < value.size()) {
      const int high = hex_value(value[i + 1]);
      const int low = hex_value(value[i + 2]);
      if (high < 0 || low < 0) throw std::runtime_error("invalid URL encoding");
      decoded.push_back(static_cast<char>((high << 4) | low));
      i += 2;
    } else {
      decoded.push_back(value[i]);
    }
  }
  return decoded;
}

std::unordered_map<std::string, std::string> parse_query(const std::string& raw) {
  std::unordered_map<std::string, std::string> result;
  std::size_t start = 0;
  while (start < raw.size()) {
    const auto end = raw.find('&', start);
    const auto part = raw.substr(start, end == std::string::npos ? std::string::npos : end - start);
    const auto eq = part.find('=');
    if (eq != std::string::npos) {
      result[url_decode(part.substr(0, eq))] = url_decode(part.substr(eq + 1));
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return result;
}

HttpRequest parse_request(const std::string& headers, const std::string& body) {
  HttpRequest request;
  request.body = body;
  std::istringstream stream(headers);
  std::string line;
  if (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream first(line);
    std::string target;
    first >> request.method >> target;
    const auto query_pos = target.find('?');
    request.path = target.substr(0, query_pos);
    if (query_pos != std::string::npos) request.query = parse_query(target.substr(query_pos + 1));
  }
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    auto key = line.substr(0, colon);
    auto value = line.substr(colon + 1);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    request.headers[std::move(key)] = std::move(value);
  }
  return request;
}

std::string status_text(int status) {
  switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 504: return "Gateway Timeout";
    default: return "OK";
  }
}

std::string serialize_response(const HttpResponse& response) {
  std::ostringstream out;
  out << "HTTP/1.1 " << response.status << " " << status_text(response.status) << "\r\n"
      << "Content-Type: " << response.content_type << "\r\n"
      << "Content-Length: " << response.body.size() << "\r\n"
      << "Connection: close\r\n\r\n" << response.body;
  return out.str();
}

}  // namespace

class HttpServer::Impl {
 public:
  explicit Impl(std::uint16_t port)
      : acceptor_(io_, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port)) {}

  void route(const std::string& path, HttpHandler handler) { routes_[path] = std::move(handler); }

  void start() {
    running_.store(true, std::memory_order_release);
    accept();
    io_.run();
  }

  void stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    asio::post(io_, [this]() {
      asio::error_code error;
      acceptor_.close(error);
      io_.stop();
    });
  }

 private:
  class Session : public std::enable_shared_from_this<Session> {
   public:
    Session(asio::ip::tcp::socket socket, const std::unordered_map<std::string, HttpHandler>& routes)
        : socket_(std::move(socket)), routes_(routes) {}

    void start() { read_headers(); }

   private:
    void read_headers() {
      auto self = shared_from_this();
      asio::async_read_until(socket_, buffer_, "\r\n\r\n",
          [self](const asio::error_code& error, std::size_t bytes) {
            if (error) return;
            if (self->buffer_.size() > kMaxRequestBytes) return self->write_error(413, "request too large");
            std::string headers(bytes, '\0');
            self->buffer_.sgetn(headers.data(), static_cast<std::streamsize>(bytes));
            self->headers_ = std::move(headers);
            std::size_t content_length = 0;
            std::istringstream stream(self->headers_);
            std::string line;
            while (std::getline(stream, line)) {
              auto colon = line.find(':');
              if (colon == std::string::npos) continue;
              auto key = line.substr(0, colon);
              std::transform(key.begin(), key.end(), key.begin(),
                             [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
              if (key == "content-length") {
                try { content_length = std::stoull(line.substr(colon + 1)); }
                catch (...) { return self->write_error(400, "invalid content-length"); }
              }
            }
            if (content_length > kMaxRequestBytes) return self->write_error(413, "request too large");
            self->read_body(content_length);
          });
    }

    void read_body(std::size_t length) {
      if (buffer_.size() >= length) return dispatch(length);
      auto self = shared_from_this();
      asio::async_read(socket_, buffer_, asio::transfer_exactly(length - buffer_.size()),
          [self, length](const asio::error_code& error, std::size_t) {
            if (!error) self->dispatch(length);
          });
    }

    void dispatch(std::size_t length) {
      std::string body(length, '\0');
      if (length > 0) buffer_.sgetn(body.data(), static_cast<std::streamsize>(length));
      HttpResponse response;
      try {
        const auto request = parse_request(headers_, body);
        const auto it = routes_.find(request.path);
        response = it == routes_.end()
            ? HttpResponse{404, "application/json", "{\"error\":\"not found\"}"}
            : it->second(request);
      } catch (const std::exception& error) {
        response = HttpResponse{500, "application/json",
            std::string("{\"error\":\"") + json_escape(error.what()) + "\"}"};
      }
      write(std::move(response));
    }

    void write_error(int status, const std::string& message) {
      write(HttpResponse{status, "application/json",
          std::string("{\"error\":\"") + json_escape(message) + "\"}"});
    }

    void write(HttpResponse response) {
      output_ = serialize_response(response);
      auto self = shared_from_this();
      asio::async_write(socket_, asio::buffer(output_),
          [self](const asio::error_code&, std::size_t) {
            asio::error_code ignored;
            self->socket_.shutdown(asio::ip::tcp::socket::shutdown_both, ignored);
            self->socket_.close(ignored);
          });
    }

    asio::ip::tcp::socket socket_;
    asio::streambuf buffer_;
    const std::unordered_map<std::string, HttpHandler>& routes_;
    std::string headers_;
    std::string output_;
  };

  void accept() {
    acceptor_.async_accept([this](const asio::error_code& error, asio::ip::tcp::socket socket) {
      if (!error) std::make_shared<Session>(std::move(socket), routes_)->start();
      if (running_.load(std::memory_order_acquire)) accept();
    });
  }

  asio::io_context io_;
  asio::ip::tcp::acceptor acceptor_;
  std::unordered_map<std::string, HttpHandler> routes_;
  std::atomic<bool> running_{false};
};

HttpServer::HttpServer(std::uint16_t port) : impl_(std::make_unique<Impl>(port)) {}
HttpServer::~HttpServer() = default;
void HttpServer::route(const std::string& path, HttpHandler handler) { impl_->route(path, std::move(handler)); }
void HttpServer::start() { impl_->start(); }
void HttpServer::stop() { impl_->stop(); }

std::string query_value(const HttpRequest& request, const std::string& key, const std::string& fallback) {
  const auto it = request.query.find(key);
  return it == request.query.end() ? fallback : it->second;
}

std::string json_escape(const std::string& input) {
  std::string escaped;
  escaped.reserve(input.size());
  for (char ch : input) {
    switch (ch) {
      case '"': escaped += "\\\""; break;
      case '\\': escaped += "\\\\"; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default: escaped += ch; break;
    }
  }
  return escaped;
}

}  // namespace aiinfra::net

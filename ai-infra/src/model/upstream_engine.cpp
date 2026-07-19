#include "aiinfra/model/inference_engine.h"

#include <asio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aiinfra::model {
namespace {

std::string env_or(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0' ? value : fallback;
}

std::size_t env_size(const char* name, std::size_t fallback) {
  try { return std::max<std::size_t>(1, std::stoull(env_or(name, ""))); }
  catch (...) { return fallback; }
}

std::string json_escape(const std::string& input) {
  std::string output;
  output.reserve(input.size() + 16);
  for (char ch : input) {
    switch (ch) {
      case '\\': output += "\\\\"; break;
      case '"': output += "\\\""; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default: output += ch; break;
    }
  }
  return output;
}

std::string extract_json_string(const std::string& body, const std::string& field) {
  const auto key = body.find("\"" + field + "\"");
  if (key == std::string::npos) return {};
  const auto colon = body.find(':', key);
  const auto quote = body.find('"', colon);
  if (quote == std::string::npos) return {};
  std::string value;
  bool escaped = false;
  for (std::size_t i = quote + 1; i < body.size(); ++i) {
    const char ch = body[i];
    if (escaped) {
      if (ch == 'n') value += '\n';
      else if (ch == 'r') value += '\r';
      else if (ch == 't') value += '\t';
      else value += ch;
      escaped = false;
    } else if (ch == '\\') escaped = true;
    else if (ch == '"') return value;
    else value += ch;
  }
  return {};
}

std::string post_upstream(const std::string& body) {
  const auto host = env_or("AIINFRA_UPSTREAM_HOST", "127.0.0.1");
  const auto port = env_or("AIINFRA_UPSTREAM_PORT", "8000");
  const auto path = env_or("AIINFRA_UPSTREAM_PATH", "/internal/infer");
  const auto secret = env_or("AIINFRA_SHARED_SECRET", "development-only");
  asio::ip::tcp::iostream stream;
  stream.expires_after(std::chrono::milliseconds(env_size("AIINFRA_BACKEND_TIMEOUT_MS", 120000)));
  stream.connect(host, port);
  if (!stream) throw std::runtime_error("upstream connection failed");
  stream << "POST " << path << " HTTP/1.1\r\nHost: " << host
         << "\r\nContent-Type: application/json\r\nX-AIInfra-Secret: " << secret
         << "\r\nConnection: close\r\nContent-Length: " << body.size()
         << "\r\n\r\n" << body << std::flush;
  std::ostringstream response_stream;
  response_stream << stream.rdbuf();
  const auto response = response_stream.str();
  const auto headers_end = response.find("\r\n\r\n");
  if (headers_end == std::string::npos || response.find(" 200 ") == std::string::npos) {
    throw std::runtime_error("upstream returned an invalid response");
  }
  return response.substr(headers_end + 4);
}

class UpstreamEngine final : public InferenceEngine {
 public:
  void load(const ModelConfig& config) override { config_ = config; loaded_ = true; }
  void unload() override { loaded_ = false; }
  bool loaded() const override { return loaded_; }

  std::vector<scheduler::InferenceResponse> infer_batch(
      const std::vector<scheduler::InferenceRequest>& requests) override {
    const auto infer_one = [this](scheduler::InferenceRequest request) {
      const auto started = scheduler::Clock::now();
      scheduler::InferenceResponse result;
      result.request_id = request.request_id;
      try {
        const auto body = "{\"model\":\"" + json_escape(config_.path) +
            "\",\"prompt\":\"" + json_escape(request.prompt) +
            "\",\"max_tokens\":" + std::to_string(request.max_tokens) + "}";
        result.text = extract_json_string(post_upstream(body), "text");
        if (result.text.empty()) throw std::runtime_error("upstream response text is empty");
        result.ok = true;
      } catch (const std::exception& error) {
        result.ok = false;
        result.error = error.what();
      }
      result.inference_latency = std::chrono::duration_cast<std::chrono::microseconds>(
          scheduler::Clock::now() - started);
      return result;
    };

    std::vector<scheduler::InferenceResponse> results;
    results.reserve(requests.size());
    const auto max_parallel = env_size("AIINFRA_BACKEND_CONCURRENCY", 4);
    for (std::size_t offset = 0; offset < requests.size(); offset += max_parallel) {
      const auto end = std::min(requests.size(), offset + max_parallel);
      std::vector<std::future<scheduler::InferenceResponse>> pending;
      for (std::size_t i = offset; i < end; ++i) {
        pending.push_back(std::async(std::launch::async, infer_one, requests[i]));
      }
      for (auto& future : pending) results.push_back(future.get());
    }
    return results;
  }

 private:
  bool loaded_{false};
  ModelConfig config_;
};

}  // namespace

std::unique_ptr<InferenceEngine> make_configured_inference_engine() {
  return std::make_unique<UpstreamEngine>();
}

}  // namespace aiinfra::model

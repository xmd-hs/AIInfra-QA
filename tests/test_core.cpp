#include "gpuforge/kernels.hpp"
#include "gpuforge/graph.hpp"
#include "gpuforge/autotune.hpp"
#include "gpuforge/memory.hpp"
#include "gpuforge/precision.hpp"
#include "gpuforge/telemetry.hpp"
#include "gpuforge/pipeline.hpp"
#include "gpuforge/planner.hpp"
#include "gpuforge/scheduler.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
using namespace gpuforge;

static void tensor_test() {
  Tensor a({2, 3}), b({3, 2});
  a.fill(1.0f); b.fill(2.0f);
  auto out = gemm(a, b);
  assert(out.at(0) == 6.0f);
  assert(compare(out, out).pass(1e-6f));
}

static void paged_cache_test() {
  PagedKV cache(1, 1, 8, 4);
  unsigned char input[20];
  for (size_t i = 0; i < sizeof(input); ++i) input[i] = static_cast<unsigned char>(i);
  cache.append(2, input, sizeof(input));
  assert(cache.bytes(2) == sizeof(input));
  assert(cache.resident(2) == 3);
  auto location = cache.locate(2, 8);
  assert(location.valid && location.offset == 0);
  unsigned char output[20] = {};
  assert(cache.read(2, 0, output, sizeof(output)) == sizeof(output));
  assert(std::memcmp(input, output, sizeof(input)) == 0);
  cache.release_sequence(2);
  assert(cache.bytes(2) == 0 && cache.resident(2) == 0);
}

static void runtime_test() {
  MemoryPool pool(4096);
  void* p = pool.allocate(128);
  assert(p != nullptr && pool.used() > 0);
  pool.release(p);
  TraceRecorder trace;
  { ScopedTrace scope(trace, "runtime_test", "unit"); }
  assert(trace.events().size() == 1);
  assert(trace.chrome_json().find("traceEvents") != std::string::npos);
}

static void boundary_test() {
  PagedKV cache(1, 1, 4, 1);
  unsigned char data[9] = {};
  cache.append(1, data, 4);
  bool exhausted = false;
  try { cache.append(1, data, 5); } catch (const std::bad_alloc&) { exhausted = true; }
  assert(exhausted);

  runtime::Scheduler scheduler(1);
  scheduler.submit({11, 2, 5, 0});
  scheduler.submit({12, 2, 5, 0});
  scheduler.cancel(11);
  assert(scheduler.pending() == 1);
  scheduler.cancel(12);
  assert(scheduler.pending() == 0);
}

static void planner_boundary_test() {
  using namespace compiler;
  TilePlanner planner;
  Schedule schedule;
  schedule.tensor_core = true;
  const Shape aligned_a{{128, 64}, ScalarType::F16, Layout::RowMajor};
  const Shape aligned_b{{64, 128}, ScalarType::F16, Layout::RowMajor};
  const auto aligned = planner.plan(aligned_a, aligned_b, schedule);
  assert(aligned.kind == KernelKind::WmmaTensorCore);

  const Shape ragged_a{{127, 63}, ScalarType::F16, Layout::RowMajor};
  const Shape ragged_b{{63, 129}, ScalarType::F16, Layout::RowMajor};
  const auto ragged = planner.plan(ragged_a, ragged_b, schedule);
  assert(ragged.kind == KernelKind::CudaTiled);
  assert(ragged.padded);
  assert(planner.workspace(ragged, ragged_a) > 0);
}

static void pipeline_report_test() {
  using namespace compiler;
  Module module;
  const int lhs = module.parameter("lhs", {{32, 16}, ScalarType::F16, Layout::RowMajor});
  const int rhs = module.parameter("rhs", {{16, 32}, ScalarType::F16, Layout::RowMajor});
  module.emit(Op::MatMul, {lhs, rhs}, {{32, 32}, ScalarType::F16});
  CompilePipeline pipeline;
  const CompileReport report = pipeline.run(module);
  const std::string text = report.text();
  assert(!report.launches.empty());
  assert(text.find("kernel=") != std::string::npos);
  assert(text.find("threads=") != std::string::npos);
}

static void attention_shape_test() {
  // Non-square query/key lengths are valid and exercise Q*K^T dimensions.
  Tensor query({2, 3}), key({4, 3}), value({4, 2});
  query.fill(1.0f);
  key.fill(1.0f);
  value.fill(2.0f);
  auto output = attention(query, key, value);
  assert(output.shape() == std::vector<size_t>({2, 2}));
  for (size_t i = 0; i < output.numel(); ++i) assert(output.at(i) == 2.0f);

  bool rejected = false;
  try {
    attention(Tensor({2, 3}), Tensor({4, 2}), Tensor({4, 2}));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

static void operator_numerics_test() {
  Tensor logits({2, 3});
  logits.at(0) = 1000.0f; logits.at(1) = 999.0f; logits.at(2) = 998.0f;
  logits.at(3) = -1000.0f; logits.at(4) = -999.0f; logits.at(5) = -998.0f;
  const Tensor probabilities = softmax(logits);
  for (size_t row = 0; row < 2; ++row) {
    float sum = 0.0f;
    for (size_t col = 0; col < 3; ++col) sum += probabilities.at(row * 3 + col);
    assert(std::abs(sum - 1.0f) < 1e-5f);
  }

  Tensor input({1, 2}), gamma({2}), beta({2});
  input.at(0) = 1.0f; input.at(1) = 3.0f;
  gamma.fill(1.0f); beta.zeros();
  const Tensor normalized = layer_norm(input, gamma, beta);
  assert(std::abs(normalized.at(0) + normalized.at(1)) < 1e-5f);
  bool rejected = false;
  try { layer_norm(input, gamma, beta, 0.0f); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}

static void randomized_operator_test() {
  // Deterministic LCG keeps this test reproducible without a test framework.
  uint32_t state = 0x12345678u;
  auto next_value = [&]() {
    state = state * 1664525u + 1013904223u;
    return (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f) - 1.0f;
  };
  Tensor lhs({5, 7}), rhs({7, 3});
  for (size_t i = 0; i < lhs.numel(); ++i) lhs.at(i) = next_value();
  for (size_t i = 0; i < rhs.numel(); ++i) rhs.at(i) = next_value();
  const Tensor actual = gemm(lhs, rhs, {3, 2, false});
  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      float expected = 0.0f;
      for (size_t k = 0; k < 7; ++k) expected += lhs.at(row * 7 + k) * rhs.at(k * 3 + col);
      assert(std::abs(actual.at(row * 3 + col) - expected) < 1e-5f);
    }
  }

  Tensor q({3, 4}), k({5, 4}), v({5, 2});
  q.fill(0.25f); k.fill(0.5f); v.fill(2.0f);
  const Tensor attended = attention(q, k, v, {64, 0.0f, true});
  for (size_t i = 0; i < attended.numel(); ++i) assert(std::abs(attended.at(i) - 2.0f) < 1e-5f);
}

static void cuda_consistency_test() {
  if (!cuda_available()) return;
  Tensor a({17, 19}), b({19, 13});
  a.fill(0.25f); b.fill(-0.5f);
  const Tensor cpu = gemm(a, b), gpu = gemm_cuda(a, b);
  assert(compare(cpu, gpu).pass(1e-3f));
  Tensor q({3, 4}), k({5, 4}), v({5, 2});
  q.fill(0.25f); k.fill(0.5f); v.fill(2.0f);
  const Tensor attended = attention_cuda(q, k, v, {64, 0.0f, true});
  assert(attended.shape() == std::vector<size_t>({3, 2}));
  for (size_t i = 0; i < attended.numel(); ++i) assert(std::abs(attended.at(i) - 2.0f) < 1e-3f);
}

static void memory_pool_concurrency_test() {
  MemoryPool pool(1 << 20);
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 4; ++worker) {
    workers.emplace_back([&pool]() {
      for (int i = 0; i < 100; ++i) {
        void* ptr = pool.allocate(64, 16);
        assert(ptr != nullptr);
        pool.release(ptr);
      }
    });
  }
  for (auto& worker : workers) worker.join();
  assert(pool.used() == 0);
  void* block = pool.allocate(1024);
  bool rejected = false;
  try { pool.allocate((1 << 20) + 1); }
  catch (const std::bad_alloc&) { rejected = true; }
  assert(rejected && pool.used() == 1024);
  pool.release(block);
  assert(pool.used() == 0);
}

static void telemetry_merge_test() {
  Metrics lhs, rhs;
  lhs.observe("latency", 2.0);
  rhs.observe("latency", 4.0);
  rhs.observe("latency", 6.0);
  lhs.merge(rhs);
  const auto summary = lhs.summary("latency");
  assert(summary.count == 3);
  assert(std::abs(summary.mean() - 4.0) < 1e-9);
  assert(summary.min == 2.0 && summary.max == 6.0);
}

static void edge_contract_test() {
  using namespace compiler;
  assert(!infer_matmul({{-1, 4}, ScalarType::F32}, {{4, 4}, ScalarType::F32}).ok);
  assert(!infer_reshape({{2, 3}, ScalarType::F32}, {-2, 3}).ok);
  runtime::Scheduler scheduler(0);
  scheduler.submit({1, 1, 2, 0});
  assert(scheduler.next().ids.size() == 1);
  TraceRecorder trace;
  { ScopedTrace scope(trace, "quoted\\name", "cat\"x"); }
  assert(trace.chrome_json().find("quoted\\\\name") != std::string::npos);
  bool overflow_rejected = false;
  try { Tensor huge({static_cast<size_t>(-1), 2}); }
  catch (const std::length_error&) { overflow_rejected = true; }
  assert(overflow_rejected);
}

static void graph_validation_test() {
  Graph graph;
  const int input = graph.input("input");
  graph.add(OpKind::Gemm, {input}, "gemm");
  assert(graph.validate());
  graph.add(OpKind::Softmax, {99}, "bad");
  std::string error;
  assert(!graph.validate(&error));
  assert(error.find("invalid input") != std::string::npos);
}

static void scheduler_phase_test() {
  runtime::Scheduler scheduler(4);
  scheduler.submit({1, 3, 6, 0});
  scheduler.submit({2, 2, 6, 2});
  const auto prefill = scheduler.next();
  assert(prefill.phase == runtime::Phase::Prefill);
  assert(prefill.ids.size() == 1 && prefill.tokens == 3);
  scheduler.complete(1, 1);
  const auto decode = scheduler.next();
  assert(decode.phase == runtime::Phase::Decode);
  assert(!decode.ids.empty());
}

static void autotune_contract_test() {
  AutoTuneConfig config;
  config.warmup = 0;
  config.iterations = 1;
  GemmAutoTuner tuner(config);
  bool rejected = false;
  try {
    Tensor lhs({1, 3}), rhs({3, 2});
    tuner.run(lhs, rhs);
  } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}

static void compiler_validation_test() {
  using namespace compiler;
  Module module;
  module.parameter("input", {{8, 8}, ScalarType::F32, Layout::RowMajor});
  // References a future/nonexistent value and must be rejected before launch
  // planning rather than producing a misleading compile report.
  module.emit(Op::Relu, {99}, {{8, 8}, ScalarType::F32, Layout::RowMajor});
  CompilePipeline pipeline;
  const CompileReport report = pipeline.run(module);
  assert(report.launches.empty());
  assert(!report.diagnostics.empty());

  Module bad_matmul;
  const int a = bad_matmul.parameter("a", {{2, 3}, ScalarType::F32, Layout::RowMajor});
  const int b = bad_matmul.parameter("b", {{4, 2}, ScalarType::F32, Layout::RowMajor});
  bad_matmul.emit(Op::MatMul, {a, b}, {{2, 2}, ScalarType::F32, Layout::RowMajor});
  const CompileReport shape_report = pipeline.run(bad_matmul);
  assert(shape_report.launches.empty());
  assert(!shape_report.diagnostics.empty());

  Module printable;
  const int p = printable.parameter("x", {{4, 4}, ScalarType::F32, Layout::RowMajor});
  printable.emit(Op::Relu, {p}, {{4, 4}, ScalarType::F32, Layout::RowMajor});
  assert(printable.print().find("relu") != std::string::npos);

  Module algebra;
  const int x = algebra.parameter("x", {{4, 4}, ScalarType::F32, Layout::RowMajor});
  const int sum = algebra.emit(Op::Add, {x, x}, {{4, 4}, ScalarType::F32, Layout::RowMajor});
  PassManager passes;
  passes.canonicalize(algebra);
  assert(algebra.node(sum).op == Op::Add);

  Module bad_arity;
  const int input = bad_arity.parameter("input", {{4, 4}, ScalarType::F32, Layout::RowMajor});
  bad_arity.emit(Op::Relu, {input, input}, {{4, 4}, ScalarType::F32, Layout::RowMajor});
  std::vector<std::string> errors;
  assert(!validate(bad_arity, &errors));
  assert(!errors.empty());

  Module dce;
  const int live_input = dce.parameter("live", {{2, 2}, ScalarType::F32, Layout::RowMajor});
  const int dead_input = dce.parameter("dead", {{2, 2}, ScalarType::F32, Layout::RowMajor});
  const int live = dce.emit(Op::Relu, {live_input}, {{2, 2}, ScalarType::F32, Layout::RowMajor});
  dce.emit(Op::Relu, {dead_input}, {{2, 2}, ScalarType::F32, Layout::RowMajor});
  dce.set_outputs({live});
  assert(PassManager().dead_code_eliminate(dce) == 2);
  assert(dce.nodes().size() == 2);
}

static void constant_folding_test() {
  using namespace compiler;
  Module module;
  const int lhs = module.constant("lhs", {{2}, ScalarType::F32}, {1.0f, -2.0f});
  const int rhs = module.constant("rhs", {{2}, ScalarType::F32}, {3.0f, 4.0f});
  const int sum = module.emit(Op::Add, {lhs, rhs}, {{2}, ScalarType::F32});
  const int relu = module.emit(Op::Relu, {sum}, {{2}, ScalarType::F32});
  PassManager passes;
  passes.constant_fold(module);
  assert(module.node(sum).op == Op::Constant);
  assert(module.node(sum).constant_data[0] == 4.0f);
  assert(module.node(sum).constant_data[1] == 2.0f);
  assert(module.node(relu).op == Op::Constant);
  assert(module.node(relu).constant_data[0] == 4.0f);
  assert(module.node(relu).constant_data[1] == 2.0f);
}

int main() {
  tensor_test();
  paged_cache_test();
  runtime_test();
  boundary_test();
  planner_boundary_test();
  pipeline_report_test();
  attention_shape_test();
  compiler_validation_test();
  operator_numerics_test();
  randomized_operator_test();
  memory_pool_concurrency_test();
  telemetry_merge_test();
  edge_contract_test();
  cuda_consistency_test();
  graph_validation_test();
  scheduler_phase_test();
  autotune_contract_test();
  constant_folding_test();
  return 0;
}

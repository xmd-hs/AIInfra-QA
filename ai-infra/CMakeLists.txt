cmake_minimum_required(VERSION 3.18)

project(native_llm_orchestrator VERSION 0.1.0 LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  asio
  URL https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-30-2.tar.gz
)
FetchContent_MakeAvailable(asio)

option(AIINFRA_ENABLE_CUDA "Build optional CUDA backend pieces" OFF)
option(AIINFRA_ENABLE_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(aiinfra
  src/concurrency/thread_pool.cpp
  src/gpu/kv_cache_manager.cpp
  src/gpu/paged_memory_pool.cpp
  src/model/model_manager.cpp
  src/model/upstream_engine.cpp
  src/net/http_server.cpp
  src/observability/gpu_probe.cpp
  src/observability/metrics.cpp
  src/scheduler/continuous_batcher.cpp
)

target_include_directories(aiinfra PUBLIC include)
target_include_directories(aiinfra PRIVATE ${asio_SOURCE_DIR}/asio/include)
target_compile_definitions(aiinfra PRIVATE ASIO_STANDALONE ASIO_NO_DEPRECATED)

target_compile_options(aiinfra PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -O3>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

target_link_libraries(aiinfra PUBLIC Threads::Threads)

if(AIINFRA_ENABLE_CUDA)
  enable_language(CUDA)
  target_compile_definitions(aiinfra PUBLIC AIINFRA_ENABLE_CUDA=1)
  target_sources(aiinfra PRIVATE src/gpu/cuda_runtime_bridge.cu)
  set_property(TARGET aiinfra PROPERTY CUDA_STANDARD 17)
endif()

add_executable(infer_server apps/infer_server.cpp)
target_link_libraries(infer_server PRIVATE aiinfra)

if(WIN32)
  target_link_libraries(aiinfra PUBLIC ws2_32 mswsock)
endif()

if(AIINFRA_ENABLE_TESTS)
  enable_testing()

  add_executable(test_mpmc_queue tests/test_mpmc_queue.cpp)
  target_link_libraries(test_mpmc_queue PRIVATE aiinfra)
  add_test(NAME test_mpmc_queue COMMAND test_mpmc_queue)

  add_executable(test_memory_pool tests/test_memory_pool.cpp)
  target_link_libraries(test_memory_pool PRIVATE aiinfra)
  add_test(NAME test_memory_pool COMMAND test_memory_pool)

  add_executable(test_kv_cache_manager tests/test_kv_cache_manager.cpp)
  target_link_libraries(test_kv_cache_manager PRIVATE aiinfra)
  add_test(NAME test_kv_cache_manager COMMAND test_kv_cache_manager)

  add_executable(test_batcher tests/test_batcher.cpp)
  target_link_libraries(test_batcher PRIVATE aiinfra)
  add_test(NAME test_batcher COMMAND test_batcher)
endif()

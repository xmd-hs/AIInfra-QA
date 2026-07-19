FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    ninja-build \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAIINFRA_ENABLE_TESTS=OFF \
  && cmake --build build --target infer_server

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app
COPY --from=build /src/build/infer_server /app/infer_server

EXPOSE 8080
ENTRYPOINT ["/app/infer_server"]

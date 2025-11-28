FROM golang:1.25-alpine AS cli-builder

COPY ./cli /home/cli

RUN cd /home/cli && go build -o /model-downloader main.go

FROM ubuntu:24.04 AS llamacpp-builder

# Install OpenCL
RUN export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    apt install --no-install-recommends -y software-properties-common; \
    apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa; \
    apt update; \
    apt install --no-install-recommends -y clinfo qcom-adreno1; \
    apt install --no-install-recommends -y cmake ninja-build curl libcurl4-openssl-dev build-essential git; \
    mkdir -p ~/dev/llm; \
    rm -f /usr/lib/libOpenCL.so; \
    ln -s /lib/aarch64-linux-gnu/libOpenCL.so.1.0.0 /usr/lib/libOpenCL.so

# Install OpenCL Headers
RUN cd ~/dev/llm; \
    git clone https://github.com/KhronosGroup/OpenCL-Headers; \
    cd OpenCL-Headers; \
    mkdir -p build && cd build; \
    cmake .. -G Ninja \
        -DBUILD_TESTING=OFF \
        -DOPENCL_HEADERS_BUILD_TESTING=OFF \
        -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF \
        -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"; \
    cmake --build . --target install; \
    # ICD Loader
    cd ~/dev/llm; \
    git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader; \
    cd OpenCL-ICD-Loader; \
    mkdir -p build && cd build; \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$HOME/dev/llm/opencl" \
        -DCMAKE_INSTALL_PREFIX="$HOME/dev/llm/opencl"; \
    cmake --build . --target install; \
    # Symlink OpenCL headers
    rm -f /usr/include/CL; \
    ln -s ~/dev/llm/opencl/include/CL/ /usr/include/CL

RUN cd ~/dev/llm; \
    git clone https://github.com/ggml-org/llama.cpp; \
    cd llama.cpp; \
    # Build master, but we can switch to a specific commit if needed
    mkdir -p build; \
    cd build; \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_OPENCL=ON; \
    ninja -j`nproc`

RUN cd ~/dev/llm/llama.cpp/build/; \
    mv bin bin-full; \
    mkdir -p bin; \
    cp bin-full/llama-run bin/; \
    cp bin-full/llama-cli bin/; \
    cp bin-full/llama-server bin/

FROM ubuntu:24.04 AS production

COPY --from=llamacpp-builder /root/dev/llm/llama.cpp/build/bin /usr/local/bin/
COPY --from=cli-builder /model-downloader /usr/local/bin/model-downloader

RUN set -ex; \
    groupadd -g 44 video || true; \
    groupadd -g 993 render || true; \
    usermod -aG video ubuntu || true; \
    usermod -aG render ubuntu || true;

RUN export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    apt install --no-install-recommends -y software-properties-common; \
    apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa; \
    apt update; \
    apt install --no-install-recommends -y clinfo qcom-adreno1 curl libgomp1; \
    apt remove -y software-properties-common; \
    apt autoremove -y; \
    apt dist-clean; \
    rm -f /usr/lib/libOpenCL.so; \
    ln -s /lib/aarch64-linux-gnu/libOpenCL.so.1.0.0 /usr/lib/libOpenCL.so

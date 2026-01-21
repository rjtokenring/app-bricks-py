FROM ubuntu:24.04 AS llamacpp-builder

# Install OpenCL
RUN export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    apt install --no-install-recommends -y software-properties-common; \
    apt-add-repository -y ppa:ubuntu-qcom-iot/qcom-ppa; \
    apt update; \
    apt install --no-install-recommends -y clinfo qcom-adreno1 cmake ninja-build curl libcurl4-openssl-dev build-essential git; \
    mkdir -p ~/dev/llm; \
    rm -f /usr/lib/libOpenCL.so; \
    ln -s /lib/aarch64-linux-gnu/libOpenCL.so.1.0.0 /usr/lib/libOpenCL.so

# Install dependencies for building llama.cpp
RUN apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    ninja-build \
    ca-certificates \
    libgomp1 \
    libssl-dev:arm64 \
    && update-ca-certificates

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
		-DCMAKE_SYSTEM_NAME=Linux \
		-DCMAKE_SYSTEM_PROCESSOR=aarch64 \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_BUILD_TYPE=Release \
		-DGGML_NATIVE=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DGGML_BACKEND_DL=ON \
        # Disabling CPU repacking for compatibility issues
		-DGGML_USE_CPU_REPACK=OFF \
		-DGGML_CPU_ALL_VARIANTS=OFF \
        # Enable OpenCL
        -DGGML_OPENCL=ON; \
    ninja -j`nproc`

RUN cd ~/dev/llm/llama.cpp/build/; \
    mv bin bin-full; \
    mkdir -p bin; \
    cp bin-full/llama-cli bin/; \
    cp bin-full/llama-server bin/; \
    cp bin-full/lib* bin/

FROM ubuntu:24.04 AS production

COPY --from=llamacpp-builder /root/dev/llm/llama.cpp/build/bin /usr/local/bin/
COPY ./run.sh /

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
    apt autoremove -y; \
    apt dist-clean; \
    rm -f /usr/lib/libOpenCL.so; \
    ln -s /lib/aarch64-linux-gnu/libOpenCL.so.1.0.0 /usr/lib/libOpenCL.so; \
    chmod +x /usr/local/bin/*; \
    chmod +x /run.sh

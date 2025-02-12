cd /URDFit/experiment/mujoco_mpc && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DCMAKE_C_COMPILER:STRING=clang-12 -DCMAKE_CXX_COMPILER:STRING=clang++-12 -DMJPC_BUILD_GRPC_SERVICE:BOOL=ON && \
    cmake --build . --config=Release && \
    cd ../python && \
    python3 setup.py install

cd ../../../ && pip install -e .
#!/bin/bash
set -e

echo "=== NEON Demo Build ==="

git submodule update --init --recursive

mkdir -p build
cd build
cmake ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo "=== Готово! Запускай: ./build/neon_demo ==="
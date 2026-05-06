#!/bin/bash
echo "=== Установка зависимостей ==="

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Mac detected"
    if ! command -v brew &>/dev/null; then
        echo "Homebrew не найден — установи с brew.sh"
        exit 1
    fi
    brew install cmake glfw
else
    echo "Linux detected"
    sudo apt update
    sudo apt install -y \
        cmake \
        build-essential \
        git \
        libgl1-mesa-dev \
        libglfw3-dev \
        libx11-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev
fi

echo "=== Готово! Запускай: ./build.sh ==="
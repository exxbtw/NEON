cmake_minimum_required(VERSION 3.10)

# 🔥 ВАЖНО: ДО project()
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES arm64)
endif()

project(NeonDemo)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(third_party/glfw)

file(GLOB IMGUI_SRC
    third_party/imgui/*.cpp
    third_party/imgui/backends/imgui_impl_glfw.cpp
    third_party/imgui/backends/imgui_impl_opengl3.cpp
)

add_executable(neon_demo
    array_processing.cpp
    ${IMGUI_SRC}
)

target_include_directories(neon_demo PRIVATE
    third_party/imgui
    third_party/imgui/backends
    third_party/glfw/include
)

# Архитектура
if(APPLE)
    # Apple Silicon — NEON уже включён по умолчанию
    target_compile_options(neon_demo PRIVATE -O3)
    message(STATUS "Apple ARM64 — NEON по умолчанию")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # Linux ARM (например Raspberry Pi)
    target_compile_options(neon_demo PRIVATE -march=armv8-a+simd -O3)
    message(STATUS "AArch64 — NEON активен")
else()
    # x86
    target_compile_options(neon_demo PRIVATE -msse4.1 -msse4.2 -O3)
    message(STATUS "x86_64 — SSE активен")
endif()

target_compile_options(neon_demo PRIVATE -O3)

# OpenGL (ПОСЛЕ создания target!)
find_package(OpenGL REQUIRED)

if(APPLE)
    target_compile_definitions(neon_demo PRIVATE GL_SILENCE_DEPRECATION)
endif()

target_link_libraries(neon_demo glfw OpenGL::GL)
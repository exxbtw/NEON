#include <arm_neon.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cassert>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifdef __APPLE__
  #include <OpenGL/gl3.h>
#elif defined(__linux__)
  #include <GLES2/gl2.h>
#else
  #include <GL/gl.h>
#endif

#include <GLFW/glfw3.h>

// ─── Скалярная версия ───
int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if      (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }
    return sum;
}

// ─── NEON версия ───
int64_t process_array_neon(const int32_t* data, size_t n) {
    int64_t sum = 0;
    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);
    const int32x4_t zero = vdupq_n_s32(0);
    size_t i = 0;

    for (; i + 7 < n; i += 8) {
        __builtin_prefetch(data + i + 16, 0, 1);
        int32x4_t vec0 = vld1q_s32(data + i);
        int32x4_t vec1 = vld1q_s32(data + i + 4);

        int32x4_t sign0 = vshrq_n_s32(vec0, 31);
        int32x4_t sign1 = vshrq_n_s32(vec1, 31);
        int32x4_t abs0  = vsubq_s32(veorq_s32(vec0, sign0), sign0);
        int32x4_t abs1  = vsubq_s32(veorq_s32(vec1, sign1), sign1);

        int32x4_t mask_pos0 = vcgtq_s32(vec0, zero);
        int32x4_t mask_neg0 = vcltq_s32(vec0, zero);
        int32x4_t mask_pos1 = vcgtq_s32(vec1, zero);
        int32x4_t mask_neg1 = vcltq_s32(vec1, zero);

        int32x4_t contrib0 = vorrq_s32(vandq_s32(vec0, mask_pos0), vandq_s32(abs0, mask_neg0));
        int32x4_t contrib1 = vorrq_s32(vandq_s32(vec1, mask_pos1), vandq_s32(abs1, mask_neg1));

        acc0 = vaddq_s64(acc0, vpaddlq_s32(contrib0));
        acc1 = vaddq_s64(acc1, vpaddlq_s32(contrib1));
    }

    for (; i + 3 < n; i += 4) {
        int32x4_t vec     = vld1q_s32(data + i);
        int32x4_t sign    = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = vsubq_s32(veorq_s32(vec, sign), sign);
        int32x4_t mask_pos = vcgtq_s32(vec, zero);
        int32x4_t mask_neg = vcltq_s32(vec, zero);
        int32x4_t contrib  = vorrq_s32(vandq_s32(vec, mask_pos), vandq_s32(abs_val, mask_neg));
        acc0 = vaddq_s64(acc0, vpaddlq_s32(contrib));
    }

    int64x2_t acc = vaddq_s64(acc0, acc1);
    // vaddvq_s64 может отсутствовать на RPi3 — используем надёжный вариант
    sum = vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1);

    for (; i < n; ++i) {
        int32_t val = data[i];
        if      (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }
    return sum;
}

// ─── Asm вставка ───
int32_t abs_barrel_asm(int32_t x) {
    int32_t result;
    __asm__ volatile (
        "asr  w2, %w1, #31  \n"
        "eor  %w0, %w1, w2  \n"
        "sub  %w0, %w0, w2  \n"
        : "=r"(result) : "r"(x) : "w2"
    );
    return result;
}

// ─── Бенчмарк ───
using Clock = std::chrono::high_resolution_clock;

template<typename Fn>
double bench(Fn fn, const int32_t* data, size_t n, int iters = 100) {
    fn(data, n);
    auto t0 = Clock::now();
    volatile int64_t sink = 0;
    for (int k = 0; k < iters; ++k) sink = fn(data, n);
    auto t1 = Clock::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// ─── Состояние UI ───
struct AppState {
    // Параметры
    int arraySize = 4 * 1024 * 1024;
    int iters = 100;

    // Результаты
    bool ran = false;
    int64_t scalar_result = 0;
    int64_t neon_result = 0;
    double scalar_ms = 0;
    double neon_ms = 0;
    double speedup = 0;
    bool results_match = false;

    // Корректность
    bool correctness_ok = false;
    int64_t corr_scalar = 0;
    int64_t corr_neon = 0;

    // ASM demo
    int asm_input = -42;
    int32_t asm_result = 0;
    bool asm_ran = false;

    // Лог
    std::vector<std::string> log;

    void addLog(const std::string& s) { log.push_back(s); }

    std::vector<float> plot_sizes;
    std::vector<float> plot_scalar;
    std::vector<float> plot_neon;
    bool plot_ready = false;
};

int main() {
    // ─── Корректность сразу при старте ───
    AppState state;

    alignas(16) int32_t small_arr[] = {3, -5, 0, 7, -2, 0, -8, 4, 0, 1};
    size_t sn = sizeof(small_arr) / sizeof(small_arr[0]);
    state.corr_scalar = process_array_scalar(small_arr, sn);
    state.corr_neon   = process_array_neon(small_arr, sn);
    state.correctness_ok = (state.corr_scalar == state.corr_neon);

    // ─── GLFW + ImGui инициализация ───
    glfwInit();

#ifdef __APPLE__
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    const char* glsl_version = "#version 150";
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    const char* glsl_version = "#version 100";
#endif

    GLFWwindow* window = glfwCreateWindow(900, 650, "ARM NEON Demo", NULL, NULL);
    if (!window) {
        std::cerr << "Ошибка создания окна" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->AddFontFromFileTTF(
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", // macOS
        18.0f,
        NULL,
        io.Fonts->GetGlyphRangesCyrillic()
    );

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // ─── Главный цикл ───
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("ARM NEON Demo", NULL,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings);

        // ── Заголовок ──
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
            "ARM NEON: обработка массива (sum of abs)");
        ImGui::Separator();

        // ── Тест корректности ──
        ImGui::Text("Тест корректности: {3,-5,0,7,-2,0,-8,4,0,1}");
        ImGui::Text("  Scalar = %lld", (long long)state.corr_scalar);
        ImGui::Text("  NEON   = %lld", (long long)state.corr_neon);
        if (state.correctness_ok)
            ImGui::TextColored(ImVec4(0.4f,1.0f,0.4f,1.0f), "  Результат: OK");
        else
            ImGui::TextColored(ImVec4(1.0f,0.2f,0.2f,1.0f), "  Результат: FAIL!");

        ImGui::Separator();

        // ── ASM barrel shifter demo ──
        ImGui::Text("ASM barrel shifter demo:");
        ImGui::SetNextItemWidth(150);
        ImGui::InputInt("Входное значение", &state.asm_input);
        ImGui::SameLine();
        if (ImGui::Button("Вычислить abs")) {
            state.asm_result = abs_barrel_asm((int32_t)state.asm_input);
            state.asm_ran = true;
        }
        if (state.asm_ran)
            ImGui::Text("  abs_barrel_asm(%d) = %d", state.asm_input, state.asm_result);

        ImGui::Separator();

        // ── Параметры бенчмарка ──
        ImGui::Text("Параметры бенчмарка:");
        ImGui::SetNextItemWidth(200);
        ImGui::SliderInt("Размер массива (x1024x1024)", &state.arraySize,
                         1 * 1024 * 1024, 16 * 1024 * 1024);
        ImGui::SetNextItemWidth(200);
        ImGui::SliderInt("Итерации", &state.iters, 10, 500);

        if (ImGui::Button("Запустить бенчмарк")) {
            state.addLog("Запуск бенчмарка...");

            int32_t* data = (int32_t*)aligned_alloc(16, state.arraySize * sizeof(int32_t));
            if (data) {
                srand(42);
                for (int k = 0; k < state.arraySize; ++k) {
                    int r = rand() % 3;
                    if      (r == 0) data[k] = 0;
                    else if (r == 1) data[k] =  (rand() % 100000) + 1;
                    else             data[k] = -(rand() % 100000) - 1;
                }

                state.scalar_result = process_array_scalar(data, state.arraySize);
                state.neon_result   = process_array_neon(data, state.arraySize);
                state.results_match = (state.scalar_result == state.neon_result);

                state.scalar_ms = bench(process_array_scalar, data, state.arraySize, state.iters);
                state.neon_ms   = bench(process_array_neon,   data, state.arraySize, state.iters);
                state.speedup   = state.scalar_ms / state.neon_ms;
                state.ran = true;

                free(data);
                state.addLog("Готово!");
            } else {
                state.addLog("Ошибка: нет памяти!");
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Построить график")) {
            state.plot_sizes.clear();
            state.plot_scalar.clear();
            state.plot_neon.clear();

            std::vector<int> sizes = {
                256 * 1024,
                512 * 1024,
                1 * 1024 * 1024,
                2 * 1024 * 1024,
                4 * 1024 * 1024,
                8 * 1024 * 1024
            };

            for (int sz : sizes) {
                int32_t* data = (int32_t*)aligned_alloc(16, sz * sizeof(int32_t));
                if (!data) continue;

                for (int i = 0; i < sz; ++i) {
                    int r = rand() % 3;
                    if      (r == 0) data[i] = 0;
                    else if (r == 1) data[i] = rand() % 100000 + 1;
                    else             data[i] = -(rand() % 100000) - 1;
                }

                double t_scalar = bench(process_array_scalar, data, sz, 50);
                double t_neon   = bench(process_array_neon,   data, sz, 50);

                state.plot_sizes.push_back((float)sz / (1024 * 1024));
                state.plot_scalar.push_back((float)t_scalar);
                state.plot_neon.push_back((float)t_neon);

                free(data);
            }

            state.plot_ready = true;
        }

        // ── Результаты ──
        if (state.ran) {
            ImGui::Separator();
            ImGui::Text("Результаты:");
            ImGui::Text("  Массив: %d элементов (%.0f МБ)",
                state.arraySize,
                state.arraySize * 4.0 / (1024 * 1024));

            ImGui::Columns(3, "results");
            ImGui::SetColumnWidth(0, 200);
            ImGui::SetColumnWidth(1, 150);
            ImGui::SetColumnWidth(2, 120);
            ImGui::Text("Реализация"); ImGui::NextColumn();
            ImGui::Text("Время");      ImGui::NextColumn();
            ImGui::Text("Ускорение");  ImGui::NextColumn();
            ImGui::Separator();

            ImGui::Text("Scalar (эталон)"); ImGui::NextColumn();
            ImGui::Text("%.3f ms", state.scalar_ms); ImGui::NextColumn();
            ImGui::Text("1.00x"); ImGui::NextColumn();

            ImGui::TextColored(ImVec4(0.4f,1.0f,0.4f,1.0f), "NEON (ARM)");
            ImGui::NextColumn();
            ImGui::TextColored(ImVec4(0.4f,1.0f,0.4f,1.0f), "%.3f ms", state.neon_ms);
            ImGui::NextColumn();
            ImGui::TextColored(ImVec4(1.0f,0.8f,0.0f,1.0f), "%.2fx", state.speedup);
            ImGui::NextColumn();
            ImGui::Columns(1);

            ImGui::Separator();
            ImGui::Text("Результат scalar: %lld", (long long)state.scalar_result);
            ImGui::Text("Результат NEON:   %lld", (long long)state.neon_result);
            if (state.results_match)
                ImGui::TextColored(ImVec4(0.4f,1.0f,0.4f,1.0f), "Результаты совпадают: OK");
            else
                ImGui::TextColored(ImVec4(1.0f,0.2f,0.2f,1.0f), "НЕСООТВЕТСТВИЕ РЕЗУЛЬТАТА!");
        }

        if (state.plot_ready) {
            ImGui::Separator();
            ImGui::Text("График: время vs размер массива (мс)");

            float max_scalar = *std::max_element(state.plot_scalar.begin(), state.plot_scalar.end());
            float max_neon   = *std::max_element(state.plot_neon.begin(), state.plot_neon.end());
            float max_val = std::max(max_scalar, max_neon);

            ImGui::PlotLines("Scalar",
                state.plot_scalar.data(),
                state.plot_scalar.size(),
                0, NULL, 0.0f, max_val, ImVec2(0,120));

            ImGui::PlotLines("NEON",
                state.plot_neon.data(),
                state.plot_neon.size(),
                0, NULL, 0.0f, max_val, ImVec2(0,120));

            ImGui::Text("Размеры (MB):");
            for (float s : state.plot_sizes) {
                ImGui::SameLine();
                ImGui::Text("%.1f", s);
            }
        }

        // ── Лог ──
        if (!state.log.empty()) {
            ImGui::Separator();
            ImGui::Text("Лог:");
            for (auto& s : state.log)
                ImGui::TextUnformatted(s.c_str());
        }

        ImGui::End();

        ImGui::Render();
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
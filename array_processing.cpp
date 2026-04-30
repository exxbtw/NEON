/**
 * Эффективная обработка массива с использованием ARM NEON и оптимизаций (C++)
 *
 *   elem > 0  → прибавить elem
 *   elem < 0  → прибавить |elem|
 *   elem == 0 → пропустить
 * Результат: int64_t
 */

#include <arm_neon.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>

// ─────────────────────────────────────────────────────────────
//  1. СКАЛЯРНАЯ ВЕРСИЯ — эталон
// ─────────────────────────────────────────────────────────────
int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if      (val > 0) sum += val;
        else if (val < 0) sum -= val;  // модуль отрицательного
    }
    return sum;
}

// ─────────────────────────────────────────────────────────────
//  2. ВЕКТОРНАЯ ВЕРСИЯ — ARM NEON
// ─────────────────────────────────────────────────────────────
int64_t process_array_neon(const int32_t* data, size_t n) {
    int64_t sum = 0;

    // Аккумуляторы int64x2 — чтобы не было переполнения int32
    // (4М элементов × до 100000 ≈ 4×10^11 >> INT32_MAX ≈ 2×10^9)
    // int64x2_t хранит 2 × int64 в одном NEON-регистре
    int64x2_t acc0 = vdupq_n_s64(0);
    int64x2_t acc1 = vdupq_n_s64(0);

    const int32x4_t zero = vdupq_n_s32(0);

    size_t i = 0;

    // ── Основной цикл: 8 элементов за итерацию (два NEON-регистра) ──
    for (; i + 7 < n; i += 8) {
        // Предзагрузка следующего блока в кэш
        __builtin_prefetch(data + i + 16, 0, 1);

        // 1. Множественная загрузка (SIMD): vld1q_s32 = 128 бит за раз
        int32x4_t vec0 = vld1q_s32(data + i);
        int32x4_t vec1 = vld1q_s32(data + i + 4);

        // 2. Безветвевое вычисление модуля (баррельный шифтер):
        //    sign = арифм. сдвиг вправо на 31 →  0x00000000 если ≥0
        //                                         0xFFFFFFFF если <0
        //    abs(x) = (x XOR sign) - sign
        int32x4_t sign0 = vshrq_n_s32(vec0, 31);   // баррельный шифтер!
        int32x4_t sign1 = vshrq_n_s32(vec1, 31);

        int32x4_t abs0 = vsubq_s32(veorq_s32(vec0, sign0), sign0);
        int32x4_t abs1 = vsubq_s32(veorq_s32(vec1, sign1), sign1);

        // 3. Условное выполнение без ветвлений (маски):
        //    vcgtq_s32 → 0xFFFFFFFF где vec > 0, иначе 0x00000000
        //    vcltq_s32 → 0xFFFFFFFF где vec < 0, иначе 0x00000000
        int32x4_t mask_pos0 = vcgtq_s32(vec0, zero);
        int32x4_t mask_neg0 = vcltq_s32(vec0, zero);
        int32x4_t mask_pos1 = vcgtq_s32(vec1, zero);
        int32x4_t mask_neg1 = vcltq_s32(vec1, zero);

        // Положительные → оставляем vec, отрицательные → abs, нули → 0
        int32x4_t contrib0 = vorrq_s32(vandq_s32(vec0, mask_pos0),
                                        vandq_s32(abs0, mask_neg0));
        int32x4_t contrib1 = vorrq_s32(vandq_s32(vec1, mask_pos1),
                                        vandq_s32(abs1, mask_neg1));

        // Накопление: расширяем int32 → int64 перед сложением.
        // vpaddlq_s32: попарно суммирует [0]+[1] и [2]+[3] → 2×int64.
        // Это и есть "горизонтальное сложение с расширением типа".
        acc0 = vaddq_s64(acc0, vpaddlq_s32(contrib0));
        acc1 = vaddq_s64(acc1, vpaddlq_s32(contrib1));
    }

    // ── Проход по 4 элемента (если n не кратно 8) ──
    for (; i + 3 < n; i += 4) {
        int32x4_t vec = vld1q_s32(data + i);

        int32x4_t sign    = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = vsubq_s32(veorq_s32(vec, sign), sign);

        int32x4_t mask_pos = vcgtq_s32(vec, zero);
        int32x4_t mask_neg = vcltq_s32(vec, zero);

        int32x4_t contrib = vorrq_s32(
            vandq_s32(vec, mask_pos),
            vandq_s32(abs_val, mask_neg)
        );

        acc0 = vaddq_s64(acc0, vpaddlq_s32(contrib));
    }

    // 4. Горизонтальное сложение: объединяем acc0 и acc1,
    //    затем vaddvq_s64 суммирует lane[0]+lane[1] → один int64
    int64x2_t acc = vaddq_s64(acc0, acc1);
    sum = vaddvq_s64(acc);
    //sum = vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1);

    // ── Скалярный хвост: остаток 0–3 элемента ──
    for (; i < n; ++i) {
        int32_t val = data[i];
        if      (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }

    return sum;
}

// ── Ассемблерная вставка: демонстрация баррельного шифтера (AArch64) ──
int32_t abs_barrel_asm(int32_t x) {
    int32_t result;
    __asm__ volatile (
        "asr  w2, %w1, #31  \n"  // w2 = sign mask (баррельный шифтер)
        "eor  %w0, %w1, w2  \n"  // result = x XOR sign
        "sub  %w0, %w0, w2  \n"  // result -= sign  → abs(x)
        : "=r"(result)
        : "r"(x)
        : "w2"
    );
    return result;
}

// ─────────────────────────────────────────────────────────────
//  ТЕСТЫ И БЕНЧМАРК
// ─────────────────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;

template<typename Fn>
double bench(Fn fn, const int32_t* data, size_t n, int iters = 100) {
    fn(data, n);  // прогрев
    auto t0 = Clock::now();
    volatile int64_t sink = 0;
    for (int k = 0; k < iters; ++k) sink = fn(data, n);
    auto t1 = Clock::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
    std::cout << "=== ARM NEON: обработка массива (sum of abs) ===\n\n";

    // ── Тест корректности ──
    // Выровненный массив (alignas(16) → vld1q_s32 без штрафа)
    alignas(16) int32_t small[] = {3, -5, 0, 7, -2, 0, -8, 4, 0, 1};
    size_t sn = sizeof(small) / sizeof(small[0]);
    // 3+5+0+7+2+0+8+4+0+1 = 30

    int64_t ref  = process_array_scalar(small, sn);
    int64_t neon = process_array_neon(small, sn);

    std::cout << "[Корректность] {3,-5,0,7,-2,0,-8,4,0,1}\n";
    std::cout << "  Scalar = " << ref  << "\n";
    std::cout << "  NEON   = " << neon << "  " << (neon == ref ? "OK" : "FAIL!") << "\n\n";

    assert(neon == ref);

    // Граничные случаи
    alignas(16) int32_t zeros[] = {0, 0, 0, 0, 0};
    assert(process_array_neon(zeros, 5) == 0);

    alignas(16) int32_t negs[] = {-1, -2, -3, -4, -5};
    assert(process_array_neon(negs, 5) == process_array_scalar(negs, 5));

    std::cout << "[Корректность] граничные случаи: OK\n\n";

    // Демонстрация asm-вставки
    std::cout << "[ASM barrel shifter demo]\n";
    std::cout << "  abs_barrel_asm(-42) = " << abs_barrel_asm(-42) << "\n";
    std::cout << "  abs_barrel_asm( 17) = " << abs_barrel_asm(17)  << "\n";
    std::cout << "  abs_barrel_asm(  0) = " << abs_barrel_asm(0)   << "\n\n";

    // ── Бенчмарк на большом массиве ──
    const size_t N = 4 * 1024 * 1024;  // 4М элементов, 16 МБ

    // 4. Выравнивание памяти: alignas(16) / aligned_alloc
    int32_t* data = (int32_t*)aligned_alloc(16, N * sizeof(int32_t));
    if (!data) { std::cerr << "OOM\n"; return 1; }

    srand(42);
    for (size_t k = 0; k < N; ++k) {
        int r = rand() % 3;
        if      (r == 0) data[k] = 0;
        else if (r == 1) data[k] =  (rand() % 100000) + 1;
        else             data[k] = -(rand() % 100000) - 1;
    }

    int64_t ref_result = process_array_scalar(data, N);
    int64_t neon_result = process_array_neon(data, N);

    if (neon_result != ref_result)
        std::cerr << "NEON: НЕСООТВЕТСТВИЕ РЕЗУЛЬТАТА!\n";

    double scalar_ms = bench(process_array_scalar, data, N);
    double neon_ms   = bench(process_array_neon,   data, N);
    double speedup   = scalar_ms / neon_ms;

    std::cout << "Массив: " << N << " элементов (16 МБ)\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::left << std::setw(20) << "Реализация"
              << std::setw(12) << "Время"
              << std::setw(10) << "Ускорение" << "\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::left  << std::setw(20) << "Scalar (эталон)"
              << std::right << std::setw(8) << std::fixed << std::setprecision(3)
              << scalar_ms << " ms"
              << std::setw(8) << "1.00x" << "\n";
    std::cout << std::left  << std::setw(20) << "NEON (ARM)"
              << std::right << std::setw(8) << std::fixed << std::setprecision(3)
              << neon_ms << " ms"
              << std::setw(7) << std::setprecision(2) << speedup << "x" << "\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << "Результат: " << ref_result << "\n";

    free(data);
    return 0;
}
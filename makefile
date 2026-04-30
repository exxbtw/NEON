CXX      := g++
SRC      := array_processing.cpp
TARGET   := array_processing

UNAME_M  := $(shell uname -m)

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra

ifeq ($(UNAME_M), arm64)
    CXXFLAGS += -march=armv8-a+simd
    $(info [Makefile] Цель: ARM64 (Mac M2) — NEON активен)
else ifeq ($(UNAME_M), aarch64)
    # Raspberry Pi 3 — тоже ARM но uname -m возвращает aarch64
    CXXFLAGS += -march=armv8-a+simd
    $(info [Makefile] Цель: AArch64 (Raspberry Pi) — NEON активен)
else
    CXXFLAGS += -msse4.1 -msse4.2 -mavx2
    $(info [Makefile] Цель: x86_64 — SSE2/AVX2 активен)
endif

# На Mac используем clang++, на Linux g++
ifeq ($(shell uname -s), Darwin)
    CXX := clang++
else
    CXX := g++
endif

.PHONY: all run asm clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $
	@echo "Скомпилировано: $(TARGET)"

run: $(TARGET)
	./$(TARGET)

asm: $(SRC)
	$(CXX) $(CXXFLAGS) -S -o $(TARGET).s $
	@echo "Листинг сохранён в $(TARGET).s"
	@echo ""
	@echo "=== Ключевые NEON-инструкции в листинге ==="
ifeq ($(UNAME_M), aarch64)
	@grep -E "ld1|shr|eor|sub\.4s|and\.16b|orr\.16b|add\.4s|addlv|prfm" $(TARGET).s | head -30 || echo "(нет совпадений)"
else ifeq ($(UNAME_M), arm64)
	@grep -E "ld1|shr|eor|sub\.4s|and\.16b|orr\.16b|add\.4s|addlv|prfm" $(TARGET).s | head -30 || echo "(нет совпадений)"
else
	@grep -E "vmovdqu|vpsrawi|vpxor|vpsubd|vpcmpgtd|vpand|vpor|vpaddd|vpaddq" $(TARGET).s | head -30 || echo "(нет совпадений)"
endif

clean:
	rm -f $(TARGET) $(TARGET).s
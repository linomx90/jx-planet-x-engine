CXX ?= g++
BUILD_DIR ?= ../build/native
TARGET := $(BUILD_DIR)/jx_bm6_native
SANITIZER_TARGET := $(BUILD_DIR)/jx_bm6_native_sanitized
SOURCE := jx_bm6_native.cpp
HEADERS := jx_bm6_types.hpp jx_bm6_integrator.hpp jx_bm6_io.hpp
STRICT_FLAGS := -std=c++20 -fno-fast-math -ffp-contract=off -Wall -Wextra -Wpedantic -Werror
RELEASE_FLAGS := -O3 -DNDEBUG
SANITIZER_FLAGS := -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined

.PHONY: all release self-test sanitizer clean

all: release

release: $(TARGET)

$(TARGET): $(SOURCE) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(STRICT_FLAGS) $(RELEASE_FLAGS) $(SOURCE) -o $(TARGET)

self-test: $(TARGET)
	$(TARGET) --self-test

sanitizer: $(SANITIZER_TARGET)
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
	UBSAN_OPTIONS=halt_on_error=1 \
	$(SANITIZER_TARGET) --self-test

$(SANITIZER_TARGET): $(SOURCE) $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(STRICT_FLAGS) $(SANITIZER_FLAGS) $(SOURCE) -o $(SANITIZER_TARGET)

clean:
	rm -rf $(BUILD_DIR)

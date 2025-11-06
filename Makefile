MAKEFLAGS += -j$(shell nproc)
CC = gcc
NVCC = nvcc
CFLAGS = -std=c23 -Wall -Wextra -Wpedantic -mavx2 -mfma
NVCCFLAGS = -std=c++17 -I/usr/include/vulkan -I/usr/include -I/usr/local/cuda/include
LDFLAGS = -L/usr/lib -L/usr/local/cuda/lib64 -lvulkan -lX11 -lcuda -lcudart -lstdc++
OPTFLAGS = -O3 

ifdef DEBUG
CFLAGS += -DDEBUG -O0 -g3
NVCCFLAGS += -DDEBUG -O0 -g
else
CFLAGS += $(OPTFLAGS)
NVCCFLAGS += $(OPTFLAGS)
endif

SRC_C := $(shell find src -name '*.c')
SRC_CU := $(shell find src -name '*.cu')
OBJ_C = $(SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)
TARGET = main

# Generate a hash of all source files to detect when files are added/removed
SRC_LIST := $(shell find src -type f \( -name '*.c' -o -name '*.cu' \) | sort)
SRC_HASH := $(shell echo "$(SRC_LIST)" | md5sum | cut -d' ' -f1)

# Default target: optimized build (just build, don't regenerate compile_commands)
all: exec

# Build everything (optimized)
exec: $(OBJ_C) $(OBJ_CU) style-check.py
	python style-check.py
	$(CC) $(OBJ_C) $(OBJ_CU) -o $(TARGET) $(LDFLAGS)

# Separate target to regenerate compile_commands.json when needed
compdb: check_src_hash
	bear -- make --always-make exec
	touch compile_commands.json

# Check if source list changed
check_src_hash:
	@if [ ! -f .src_hash ] || [ "$$(cat .src_hash)" != "$(SRC_HASH)" ]; then \
		echo "$(SRC_HASH)" > .src_hash; \
	fi

# Debug build: disable optimizations, add -g
debug:  
	$(MAKE) DEBUG=1 clean
	$(MAKE) DEBUG=1 exec
	@echo "\033[0;32mSUCCESS\033[0m"

# Optional: regenerate resources manually
res: src/res/res.h

src/res/res.h:
	python src/res/resgen.py

# Object rules
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Run depends on exec
run: exec
	./$(TARGET)

clean:
	rm -f $(OBJ_C) $(OBJ_CU) $(TARGET) compile_commands.json .src_hash

.PHONY: all exec debug run clean res compdb check_src_hash
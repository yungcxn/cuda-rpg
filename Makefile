MAKEFLAGS += -j$(shell nproc)

CC = gcc
NVCC = nvcc
CFLAGS = -std=c23 -Wall -Wextra -Wpedantic -O3 -mavx2 -mfma
NVCCFLAGS = -std=c++17 -I/usr/include/vulkan -I/usr/include -I/usr/local/cuda/include -O3
LDFLAGS = -L/usr/lib -lvulkan -lX11 -lcuda -lcudart

ifdef DEBUG
CFLAGS += -DDEBUG -g
NVCCFLAGS += -DDEBUG -g
endif

SRC_C := $(shell find src -name '*.c')
SRC_CU := $(shell find src -name '*.cu')
OBJ_C = $(SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)
TARGET = main

# Generate a hash of all source files to detect when files are added/removed
SRC_LIST := $(shell find src -type f \( -name '*.c' -o -name '*.cu' \) | sort)
SRC_HASH := $(shell echo "$(SRC_LIST)" | md5sum | cut -d' ' -f1)

# Default target: build everything and ensure compile_commands.json is up-to-date
all: compile_commands.json exec

# Rebuild compile_commands.json only if source list changed
compile_commands.json: .src_hash
	bear -- make --always-make exec
	touch compile_commands.json

.src_hash:
	@echo "$(SRC_HASH)" > .src_hash

# Build everything without running resgen.py automatically
exec: $(OBJ_C) $(OBJ_CU) style-check.py
	python style-check.py
	$(NVCC) $(NVCCFLAGS) $(OBJ_C) $(OBJ_CU) -o $(TARGET) $(LDFLAGS)

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

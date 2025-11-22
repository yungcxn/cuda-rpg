MAKEFLAGS += -j$(shell nproc)
CC = gcc
NVCC = nvcc
CFLAGS = -std=c23 -Wall -Wextra -Wpedantic -mavx2 -mfma
NVCCFLAGS = -std=c++17 -I/usr/include/vulkan -I/usr/include -I/usr/local/cuda/include
LDFLAGS = -L/usr/lib -L/usr/local/cuda/lib64 -lncurses -lvulkan -lX11 -lcuda -lcudart -lstdc++
OPTFLAGS = -O3 
EVALFLAGS = -O0 -g -G -lineinfo

ifdef EVAL
	CFLAGS += -DDEBUG $(EVALFLAGS)
	NVCCFLAGS += -DDEBUG $(EVALFLAGS)
else ifdef DEBUG
	CFLAGS += -DDEBUG $(OPTFLAGS)
	NVCCFLAGS += -DDEBUG $(OPTFLAGS)
else
	CFLAGS += $(OPTFLAGS)
	NVCCFLAGS += $(OPTFLAGS)
endif



SRC_C := $(shell find src -name '*.c')
SRC_CU := $(shell find src -name '*.cu')
OBJ_C = $(SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)
TARGET = main

SRC_LIST := $(shell find src -type f \( -name '*.c' -o -name '*.cu' \) | sort)
SRC_HASH := $(shell echo "$(SRC_LIST)" | md5sum | cut -d' ' -f1)

all: exec

exec: $(OBJ_C) $(OBJ_CU) style-check.py
	python style-check.py
	$(CC) $(OBJ_C) $(OBJ_CU) -o $(TARGET) $(LDFLAGS)

compdb: check_src_hash
	bear -- make --always-make exec
	touch compile_commands.json

check_src_hash:
	@if [ ! -f .src_hash ] || [ "$$(cat .src_hash)" != "$(SRC_HASH)" ]; then \
		echo "$(SRC_HASH)" > .src_hash; \
	fi

debug:  
	$(MAKE) DEBUG=1 clean
	$(MAKE) DEBUG=1 exec
	@echo "\033[0;32mSUCCESS\033[0m"

eval:
	$(MAKE) EVAL=1 clean
	$(MAKE) EVAL=1 exec
	@echo "\033[0;32mSUCCESS\033[0m"

res: src/res/res.h

src/res/res.h:
	python src/res/resgen.py

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

debug-run:
	$(MAKE) DEBUG=1 clean exec
	./$(TARGET)

run:
	./$(TARGET)

clean:
	rm -f $(OBJ_C) $(OBJ_CU) $(TARGET) compile_commands.json .src_hash

.PHONY: all exec debug run clean res compdb check_src_hash
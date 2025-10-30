MAKEFLAGS += -j$(shell nproc)

CC = gcc
NVCC = nvcc
CFLAGS = -std=c23 -Wall -Wextra -Wpedantic -O3 -mavx2 -mfma
NVCCFLAGS = -rdc=true -I/usr/include/vulkan -I/usr/include -I/usr/local/cuda/include -O3
LDFLAGS = -L/usr/lib -lvulkan -lX11 -lcuda -lcudart

ifdef DEBUG
CFLAGS += -DDEBUG -g
NVCCFLAGS += -DDEBUG -g
endif

SRC_C = $(wildcard src/*.c) $(wildcard src/world/*.c) $(wildcard src/world/*.c) $(wildcard src/render/*.c)
SRC_CU = $(wildcard src/render/*.cu) $(wildcard src/render/util/*.cu) 
OBJ_C = $(SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)
TARGET = main

# Default target just builds
all: exec

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
	rm -f $(OBJ_C) $(OBJ_CU) $(TARGET)

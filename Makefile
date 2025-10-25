CC = gcc
NVCC = nvcc
CFLAGS = -std=c23 -Wall -Wextra -Wpedantic -O3 -mavx2 -mfma
NVCCFLAGS = -rdc=true -I/usr/include/vulkan -I/usr/include -I/usr/local/cuda/include -O3
LDFLAGS = -L/usr/lib -lvulkan -lX11 -lcuda -lcudart

ifdef DEBUG
CFLAGS += -DDEBUG -g
NVCCFLAGS += -DDEBUG -g
endif

SRC_C = src/main.c src/live.c src/key.c src/world/ecs.c src/world/world.c
SRC_CU = src/render/render.cu src/render/kernel.cu src/render/tex.cu src/render/tileinfo.cu src/render/spriteinfo.cu 
OBJ_C = $(SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)
TARGET = main

all: $(TARGET)

# Make main depends on objects + res.h
$(TARGET): $(OBJ_C) $(OBJ_CU)

# Make res.h by running your python script
src/res/res.h: src/res/resgen.py
	python src/res/resgen.py

# Make C objects depend on res.h
%.o: %.c src/res/res.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link
$(TARGET): style-check.py src/res/res.h
	python style-check.py
	$(NVCC) $(NVCCFLAGS) $(OBJ_C) $(OBJ_CU) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJ_C) $(OBJ_CU) $(TARGET) src/res/res.h

run: $(TARGET)
	./$(TARGET)

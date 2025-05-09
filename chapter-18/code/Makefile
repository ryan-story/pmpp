NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_60 # Adjust the architecture as needed for your GPU

# Output executable
TARGET = energy_benchmark

# Object files
OBJS = benchmark.o cenergy.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

benchmark.o: benchmark.cu benchmark.h cenergy.h
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

cenergy.o: cenergy.cu cenergy.h
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = $(OUT_DIR)/cudaMatch
OBJ	        = $(OUT_DIR)/cudaMatch.o $(OUT_DIR)/support.o
OUT_DIR		= ./build

default: $(OUT_DIR) $(EXE)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/cudaMatch.o: cudaMatch.cu kernel.cu support.h
	$(NVCC) -c -o $@ cudaMatch.cu $(NVCC_FLAGS)

$(OUT_DIR)/support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf $(OUT_DIR)

ARCH = 30

CUDA_INCLUDE := /usr/local/cuda/include
CUDA_LIB := /usr/local/cuda/lib64

NVCCFLAGS := -O3 -arch=sm_$(ARCH) -Xcompiler -Wall

SRCDIR := src
INCDIR := include
BINDIR := bin
OBJDIR := obj
BIN_COMP := $(BINDIR)/compress
BIN_DECOMP := $(BINDIR)/decompress
TEMPDIR := temp

NVCC = nvcc

COMMON_SRCS := $(SRCDIR)/huffman.cpp $(SRCDIR)/timer.cpp
COMPRESS_SRCS := $(SRCDIR)/compress.cpp
COMPRESS_CU_SRCS := $(SRCDIR)/compress_utils.cu
DECOMPRESS_SRCS := $(SRCDIR)/decompress.cpp
DECOMPRESS_CU_SRCS := $(SRCDIR)/decompress_utils.cu

COMMON_OBJS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(COMMON_SRCS))
COMPRESS_OBJS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(COMPRESS_SRCS))
DECOMPRESS_OBJS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(DECOMPRESS_SRCS))
COMRPESS_CU_OBJS := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(COMPRESS_CU_SRCS))
DECOMPRESS_CU_OBJS := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(DECOMPRESS_CU_SRCS))
INCLUDES := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.cuh)

.PHONY: all clean test
all: clean $(COMMON_OBJS) $(BIN_COMP) $(BIN_DECOMP)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(INCLUDES)
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`;fi
	$(CC) $(CFLAGS) -I $(CUDA_INCLUDE) -I $(CUDA_LIB) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(INCLUDES)
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`;fi
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(BIN_COMP): $(COMMON_OBJS) $(COMRPESS_CU_OBJS) $(COMPRESS_OBJS)
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`;fi
	$(NVCC) $(NVCCFLAGS) -o $@ $^
	
$(BIN_DECOMP): $(COMMON_OBJS) $(DECOMPRESS_CU_OBJS) $(DECOMPRESS_OBJS)
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`;fi
	$(NVCC) $(NVCCFLAGS) -o $@ $^
	

clean:
	rm -rf $(OBJDIR)/*.o $(BIN_COMP) $(BIN_DECOMP)

test:
	mkdir -p $(TEMPDIR)
	base64 /dev/urandom | head -c 50000000 > $(TEMPDIR)/testFile.txt
	$(BIN_COMP) $(TEMPDIR)/testFile.txt
	$(BIN_DECOMP) $(TEMPDIR)/compressed_output.bin
	diff $(TEMPDIR)/decompressed_output $(TEMPDIR)/testFile.txt

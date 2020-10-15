all: compress decompress
	@echo "Done"

compress: clean_compress
	nvcc -O3 -rdc=true -arch=sm_30 -Xcompiler -Wall -o compress huffman.cu compressKernel.cu compress.cu

decompress: clean_decompress
	nvcc -O3 -rdc=true -arch=sm_30 -Xcompiler -Wall -o decompress huffman.cu decompressKernel.cu decompress.cu

clean_compress:
	rm -f compress

clean_decompress:
	rm -f decompress

clean: clean_compress clean_decompress

test:
	base64 /dev/urandom | head -c 1000 > testFile.txt
	./compress testFile.txt
	./decompress compressed_output.bin
	diff decompressed_output testFile.txt
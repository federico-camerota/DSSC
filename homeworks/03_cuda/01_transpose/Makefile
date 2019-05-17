CC=nvcc
CCFLAGS=-Wall -Wextra

fast_transpose=transpose.x
TRANSPOSE_FILES=fast_transpose.cu

all: ${fast_transpose}

transpose: ${fast_transpose}

${fast_transpose} : ${TRANSPOSE_FILES}
	${CC} -o $@ $^

clean:
	rm *.x

.PHONY: all clean

CC=mpicc
CCFLAGS=-Wall -Wextra
SOURCES=identity_mat.c

all: identity_matrix.x

identity_matrix.x: ${SOURCES}
	${CC} -o $@ $^ ${CCFLAGS}

clean:
	rm *.x

.PHONY: all clean


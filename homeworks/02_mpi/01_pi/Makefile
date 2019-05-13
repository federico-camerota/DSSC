CC=mpicc
CCFLAGS= -Wall -Wextra
SOURCES= pi_mpi.c

all: mpi_pi.x

mpi_pi.x : ${SOURCES}
	${CC} -o $@ $^ ${CCFLAGS} 

clean: 
	rm *.x

.PHONY: all clean

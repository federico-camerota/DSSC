CC=mpicc
CCFLAGS= -Wall -Wextra
SOURCES= ring_exchange.c

all: multiple_data_ring.x single_data_ring.x

multiple_data_ring.x : ${SOURCES}
	${CC} -o $@ $^ ${CCFLAGS} -DMULTIPLE_DATA_RING

single_data_ring.x : ${SOURCES}
	${CC} -o $@ $^ ${CCFLAGS}

clean: 
	rm *.x

.PHONY: all clean

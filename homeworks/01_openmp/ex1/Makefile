CC=gcc
CCFLAGS=-Wall -Wextra
CCOMP = -fopenmp

SERIAL=serial_pi.x
PARALLEL=parallel_pi.x

all: $(SERIAL) $(PARALLEL)

serial: $(SERIAL)

$(SERIAL): $(SERIAL:.x=.c) 
	$(CC) -o $@ $^ $(CCFLAGS)

parallel: $(PARALLEL)

$(PARALLEL): $(PARALLEL:.x=.c) 
	$(CC) -o $@ $^ $(CCFLAGS) $(CCOMP)

clean:
	rm *.x

.PHONY: clean

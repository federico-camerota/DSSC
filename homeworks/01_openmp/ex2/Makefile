CC=gcc
CCFLAGS=-Wall -Wextra
CCOMP = -fopenmp

SERIAL=serial_scheduling.x
PARALLEL=parallel_scheduling.x

all: $(SERIAL) $(PARALLEL)

serial: $(SERIAL)

$(SERIAL): $(SERIAL:.x=.c) loop_schedule.c
	$(CC) -o $@ $^ $(CCFLAGS)

parallel: $(PARALLEL)

$(PARALLEL): $(PARALLEL:.x=.c) loop_schedule.c
	$(CC) -o $@ $^ $(CCFLAGS) $(CCOMP)

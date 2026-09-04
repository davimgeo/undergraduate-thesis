CC = gcc

FLAGS = -std=c99 -fopenmp -mavx2 -mfma -O3 -march=native

INCLUDE = -Iinclude -Ilibs/include -Iconfig

PYTHON_INCLUDE = $(shell python3-config --includes)
NUMPY_INCLUDE = $(shell python3 -c 'import numpy; print(numpy.get_include())')
PYTHON_LIBS = $(shell python3-config --embed --ldflags)

LIBS = -lfftw3f -lm

#MAIN = main.c
MAIN = smooth.c

#SRCS = $(shell find . -name "*.c")
SRCS = $(wildcard config/*.c) $(wildcard src/*.c) src/plot/plot.c

run:
	$(CC) $(FLAGS) \
	$(INCLUDE) \
	$(PYTHON_INCLUDE) \
	-I$(NUMPY_INCLUDE) \
	$(MAIN) $(SRCS) \
	-Llibs -lprop \
	$(LIBS) \
	$(PYTHON_LIBS) \
	-o run.out
	./run.out
	#$(MAKE) clean

clean:
	rm -f run.out

plot:
	$(MAKE) run
	python3 plots/plot_1d.py

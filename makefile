CC = gcc

FLAGS = -std=c99 -Wall -fopenmp -mavx2 -mfma -O3 -march=native

INCLUDE = -Iinclude -Ilibs/include -Iconfig                   \
          -I/usr/include/python3.12                           \
          -I/usr/lib/python3/dist-packages/numpy/core/include

LIBS = -lfftw3f -lm -lpython3.12

MAIN = main.c
#MAIN = tests/test_rickers.c
SRCS = $(wildcard config/*.c) $(wildcard src/*.c)

run:
	$(CC) $(FLAGS) $(INCLUDE) \
		$(MAIN) $(SRCS) \
		-Llibs -lprop \
		$(LIBS) \
		-o run.out
	./run.out
	$(MAKE) clean

clean:
	rm -f run.out

plot:
	$(MAKE) run
	python3 plots/plot_1d.py




CC = gcc

FLAGS = -std=c99 -Wall -fopenmp -mavx2 -mfma -O3 -march=native

INCLUDE = -Iinclude -Ilibs/include -Iconfig

LIBS = -lfftw3f -lm

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

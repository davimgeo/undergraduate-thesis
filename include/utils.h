#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <time.h>

#define PROFILE_BEGIN()                                   \
    struct timespec start, end;                           \
    clock_gettime(CLOCK_MONOTONIC, &start)                \

#define PROFILE_END() do {                                \
    clock_gettime(CLOCK_MONOTONIC, &end);                 \
    double elapsed = (end.tv_sec - start.tv_sec) +        \
                     (end.tv_nsec - start.tv_nsec) / 1e9; \
    printf("Elapsed: %.4f seconds\n", elapsed);           \
} while (0)

static inline void progress_bar(int counter, int total_counter)
{
  printf("Progress: %.1f%%\n", 100.0f * (float)(counter + 1) / total_counter);
  printf("\e[1;1H\e[2J");
}

static inline float* magnitude(const float complex* arr, int height, int width)
{
  float* mag = (float*)malloc(sizeof(float*) * height * width);

  for(int i = 0; i < height; i++) {
    for(int j = 0; j < width; j++) {
      int idx = i * width + j;

      mag[idx] = fabs(arr[idx]);
    }
  }

  return mag;
}

static inline float complex* conjugate1d(const float complex* A, int size)
{
  float complex* result = malloc(sizeof(float complex) * size);

  for (int i = 0; i < size; i++) 
    result[i] = conj(A[i]);

  return result;
}

static inline float complex* conjugate2d(
  const float complex* A, int rowsA, int colsA
)
{
  int size = rowsA * colsA;
  float complex* result = malloc(sizeof(float complex) * size);

  for (int i = 0; i < size; i++) {
    result[i] = conj(A[i]);
  }

  return result;
}

static inline void normalize(float* result, int result_size)
{
  float min = result[0], max = result[0];
  for (int i = 1; i < result_size; i++)
  {
    if (result[i] < min) min = result[i];
    if (result[i] > max) max = result[i];
  }

  float range = max - min;
  if (range > 0.0f)
  {
    for (int i = 0; i < result_size; i++)
      result[i] = (result[i] - min) / range;
  }
}

static inline void print1D(float* arr, int size)
{
  for (int i = 0; i < size; ++i) 
    printf("\n%f ", arr[i]);
}

#endif // UTILS_H

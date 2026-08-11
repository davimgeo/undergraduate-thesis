#include <stdlib.h>
#include <complex.h>

#define PI 3.14159265358979323846f

float complex* computeDFT(int N, float* arr, float dt)
{
  float complex* DFT = malloc(sizeof(float complex) * N);

  #pragma omp parallel for schedule(static)
  for (int k = 0; k < N; k++)
  {
    DFT[k] = 0.0f + 0.0f * I;

    for (int n = 0; n < N; n++)
    {
      float angle = -2.0f * PI * k * n / N;

      DFT[k] += arr[n] * cexpf(I * angle);
    }
  }

  return DFT;
}

float* computeIFFT(int N, float complex* X)
{
  float* x = malloc(sizeof(float) * N);

  #pragma omp parallel for schedule(static)
  for (int n = 0; n < N; n++)
  {
    x[n] = 0.0f;

    for (int k = 0; k < N; k++)
    {
      float angle = 2.0f * PI * k * n / N;

      float complex arg = X[k] * cexpf(I * angle) / (float)N;

      x[n] += crealf(arg);
    }
  }

  return x;
}

float* dftshift(float* x, int N)
{
  float* y = malloc(sizeof(float) * N);

  int shift = N / 2;

  for (int i = 0; i < N; i++)
    y[i] = x[(i + shift) % N];

  return y;
}

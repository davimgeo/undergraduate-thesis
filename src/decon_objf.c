#include <complex.h>
#include <stdlib.h>

#include "fft.h"
#include "utils.h"

#include "plot.h"

static int initialized = 0;

static float get_epsilon(float complex*arr1, float complex*arr2, int size)
{
  float max = 0.0f;
  for(int i = 0; i < size; i++) 
  {
    float temp = cabsf(arr1[i] * arr2[i]);
    if (temp > max) max = temp;
  }
  return 0.01f * max;
}

static float* get_penalty(int nt, float dt, float t0)
{
  float* P = malloc(nt * sizeof(*P));

  for (int i = 0; i < nt; ++i)
  {
    int lag_index = (i <= nt / 2) ? i : i - nt;

    float tau = lag_index * dt / 2.0f;

    P[i] = (fabsf(tau) <= t0) ? tau : 0.0f;
  }

  return P;
}

float* get_d_1d(float* dcalc, float* dobs, float dt, int nt)
{
  float complex* fft_u_o = get_fft_1d(dobs, nt);
  float complex* fft_u_s = get_fft_1d(dcalc, nt);
  float complex* C_u_o = conjugate1d(fft_u_o, nt);

  float complex* d = malloc((size_t)nt * sizeof(float complex));

  float epsilon = get_epsilon(C_u_o, fft_u_o, nt);

  for (int i = 0; i < nt; ++i)
  {
    float complex numerator = C_u_o[i] * fft_u_s[i];

    float complex denominator = C_u_o[i] * fft_u_o[i] + epsilon;

    d[i] = numerator / denominator;
  }

  float* result = get_ifft_1d(d, nt);

  free(fft_u_o);
  free(fft_u_s);
  free(C_u_o);
  free(d);

  return result;
}

float get_decon_1d(float *dcalc, float *dobs, float dt, int nt, float t0)
{
  float result = 0.0f;

  float* d = get_d_1d(dcalc, dobs, dt, nt);
  float* P = get_penalty(nt, dt, t0);

  float w = 0.0f;

  if(!initialized)
  {
    //plot1d(P, nt);
    //plot1d(d, nt);
    initialized = 1;
  }

  for (int tau = 0; tau < nt; ++tau) 
  {
    float pc = P[tau] * d[tau];

    w      += d[tau] * d[tau];
    result += pc * pc;
  }

  free(d);
  free(P);

  return 0.5f * (result / w);
}


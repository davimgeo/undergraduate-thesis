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
  float* P = malloc(nt * sizeof(float));

  for (int i = 0; i < nt; ++i)
  {
    float tau;

    if (i <= nt / 2)
      tau = i * dt;
    else
      tau = (i - nt) * dt;

    if (fabsf(tau) <= t0)
      P[i] = tau;
    else
      P[i] = 0.0f;
  }

  return P;
}


static float* get_d_1d(
    float* u_s,
    float* u_o,
    float dt,
    int nt
)
{
  float complex* fft_u_o = get_fft_1d(u_o, nt);
  float complex* fft_u_s = get_fft_1d(u_s, nt);
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

float get_decon_1d(float *u_s, float *u_o, float dt, int nt, float t0)
{
  float result = 0.0f;

  float* d = get_d_1d(u_s, u_o, dt, nt);
  float* P = get_penalty(nt, dt, t0);

  if(!initialized)
  {
    //plot1d(P, nt);
    //plot1d(d, nt);
    initialized = 1;
  }

  for (int tau = 0; tau < nt; ++tau) 
  {
    float pc = P[tau] * d[tau];

    result += pc * pc;
  }

  free(d);
  free(P);

  return (-0.5f * result);
}

static float* get_d_2d(
    float* u_s,
    float* u_o,
    float dt,
    int nt,
    int nrec
)
{
  float* result = malloc((size_t)nt * nrec * sizeof(float));
  if (result == NULL) return NULL;

  float* trace_s = malloc((size_t)nt * sizeof(float));

  float* trace_o = malloc((size_t)nt * sizeof(float));

  for (int rec = 0; rec < nrec; ++rec)
  {
    for (int t = 0; t < nt; ++t)
    {
      trace_s[t] = u_s[t * nrec + rec];
      trace_o[t] = u_o[t * nrec + rec];
    }

    float* d = get_d_1d(
        trace_s,
        trace_o,
        dt,
        nt
    );

    for (int tau = 0; tau < nt; ++tau)
      result[tau * nrec + rec] = d[tau];

    free(d);
  }

  free(trace_s);
  free(trace_o);

  return result;
}

float get_decon_2d(float *u_s, float *u_o, float dt, int nt, int nrec, float t0)
{
  float result = 0.0f;

  float* d = get_d_2d(u_s, u_o, dt, nt, nrec);
  float* P = get_penalty(nt, dt, t0);

  if(!initialized)
  {
    //plot1d(P, nt);
    //plot1d(d, nt);
    initialized = 1;
  }

  for (int tau = 0; tau < nt; ++tau)
  {
    for (int j = 0; j < nrec; ++j)
    {
      int idx = tau * nrec + j;

      float pc = P[tau] * d[idx];

      result += pc * pc;
    }
  }

  free(d);
  free(P);

  return (0.5f * result);
}

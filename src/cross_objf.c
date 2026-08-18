#include <complex.h>
#include <stdlib.h>

#include "fft.h"
#include "utils.h"

#include "plot.h"

static int initialized = 0;

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

static float* get_c_1d(float* u_s, float* u_o, float dt, int nt)
{
  // IFFT(conj(A) * B)
  float complex* fft_u_s = get_fft_1d(u_s, nt);
  float complex* C_u_s = conjugate1d(fft_u_s, nt); 
  float complex* Im_u_o = get_fft_1d(u_o, nt);

  // cross correlation
  float complex* cross = malloc(sizeof(float complex) * nt);
  for (int i = 0; i < nt; i++) cross[i] = C_u_s[i] * Im_u_o[i];

  float* result = get_ifft_1d(cross, nt);

  free(fft_u_s); free(C_u_s); free(Im_u_o); free(cross);

  return result;
}

float get_cross_1d(float *u_s, float *u_o, float dt, int nt, float t0)
{
  // H_cor = 0.5 * sum(P(tau) * c)
  float result = 0.0f;

  float* c = get_c_1d(u_s, u_o, dt, nt);
  float* P = get_penalty(nt, dt, t0);

  if(!initialized)
  {
    //plot1d(P, nt);
    //plot1d(c_shift, nt);
    initialized = 1;
  }

  for (int tau = 0; tau < nt; ++tau) 
  {
    float pc = P[tau] * c[tau];

    result += pc * pc;
  }

  free(c); free(P);

  return (-0.5f * result);
}

float* get_c_2d(
    float* u_s,
    float* u_o,
    float dt,
    int nt,
    int nrec
)
{
  float* result = malloc((size_t)nt * nrec * sizeof(float));
  if (result == NULL) return NULL;

  float* trace_s = (float*)malloc(nt * sizeof(float));
  float* trace_o = (float*)malloc(nt * sizeof(float));

  for (int rec = 0; rec < nrec; ++rec)
  {

    for (int t = 0; t < nt; ++t)
    {
      trace_s[t] = u_s[t * nrec + rec];
      trace_o[t] = u_o[t * nrec + rec];
    }

    float* c = get_c_1d(
        trace_s,
        trace_o,
        dt,
        nt
    );

    for (int tau = 0; tau < nt; ++tau) result[tau * nrec + rec] = c[tau];

    free(c);
  }

  free(trace_s);
  free(trace_o);

  return result;
}

float get_cross_2d(float *u_s, float *u_o, float dt, int nt, int nrec, float t0)
{
  // H_cor = 0.5 * sum(P(tau) * c)
  float result = 0.0f;

  float* c = get_c_2d(u_s, u_o, dt, nt, nrec);
  float* P = get_penalty(nt, dt, t0);

  if(!initialized)
  {
    //plot1d(P, nt);
    //plot2d(c, nt, nrec);
    initialized = 1;
  }

  for (int tau = 0; tau < nt; ++tau)
  {
    for (int j = 0; j < nrec; ++j)
    {
      int idx = tau * nrec + j;

      float pc = P[tau] * c[idx];

      result += pc * pc;
    }
  }

  free(c); free(P);

  return (0.5f * result);
}

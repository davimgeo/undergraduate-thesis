#include <complex.h>
#include <stdlib.h>

#include "fft.h"
#include "utils.h"

#include "plot.h"

static int initialized = 0;

static float* get_penalty(int nt, float dt, float t0)
{
  float* P = (float*)malloc(nt * sizeof(float));

  for (int i = 0; i < nt; ++i) 
  {
    float tau = (i - (float)nt/2) * dt;

    if(fabs(tau) <= t0) {
      P[i] = tau;
    } else {
      P[i] = 0.0f;
    }
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

static float* get_c_2d(float* u_s, float* u_o, float dt, int nt, int nrec)
{
  float complex* fft_u_s = get_fft_2d(u_s, nt, nrec);
  float complex* C_u_s   = conjugate2d(fft_u_s, nt, nrec);
  float complex* Im_u_o  = get_fft_2d(u_o, nt, nrec);

  float complex* cross = malloc(sizeof(float complex) * nt * nrec);

  for (int i = 0; i < nt * nrec; i++) cross[i] = C_u_s[i] * Im_u_o[i];

  float* result = get_ifft_2d(cross, nt, nrec);

  free(fft_u_s); free(C_u_s); free(Im_u_o); free(cross);

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

  return (-0.5f * result);
}

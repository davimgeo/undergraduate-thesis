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

static float* get_d_1d(
  float* u_s, float* u_o,
  float dt, int nt
)
{
  // IFFT( (conj(A) * B) / (conj(A) * A + eps) )
  
  float complex* C_u_o = conjugate1d(get_fft_1d(u_o, nt), nt); 
  float complex* Im_u_o = get_fft_1d(u_o, nt);
  float complex* Im_u_s = get_fft_1d(u_s, nt);

  float complex* d = malloc(sizeof(float complex) * nt);

  float epsilon = get_epsilon(C_u_o, Im_u_o, nt);
  for (int i = 0; i < nt; i++)
  {
    d[i] = (C_u_o[i] * Im_u_s[i]) / ((C_u_o[i] * Im_u_o[i]) + epsilon);
  }

  float* result = get_ifft_1d(d, nt);

  free(C_u_o); free(Im_u_o); free(Im_u_s); free(d);

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

static float* get_d_2d(float* u_s, float* u_o, float dt, int nt, int nrec)
{
  // IFFT( (conj(A) * B) / (conj(A) * A + eps) )
  
  float complex* fft_u_o = get_fft_2d(u_o, nt, nrec);
  float complex* C_u_o   = conjugate2d(fft_u_o, nt, nrec);
  float complex* Im_u_o  = get_fft_2d(u_o, nt, nrec);
  float complex* Im_u_s  = get_fft_2d(u_s, nt, nrec);

  float complex* d = malloc(sizeof(float complex) * nt * nrec);

  float epsilon = get_epsilon(C_u_o, Im_u_o, nt * nrec);
  for (int i = 0; i < nt * nrec; i++)
    d[i] = (C_u_o[i] * Im_u_s[i]) / ((C_u_o[i] * Im_u_o[i]) + epsilon);

  float* result = get_ifft_2d(d, nt, nrec);

  free(C_u_o); free(Im_u_o); free(Im_u_s); free(d);

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

  return (-0.5f * result);
}

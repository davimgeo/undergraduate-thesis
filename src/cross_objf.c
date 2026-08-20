#include <complex.h>
#include <math.h>
#include <stdlib.h>

#include "fft.h"
#include "plot.h"

static int initialized = 0;

static float* get_penalty(float tau0, int nt, float dt)
{
  float* P = malloc(nt * sizeof(*P));

  for (int itau = 0; itau < nt; ++itau)
  {
    int lag_index = (itau <= nt / 2) ? itau : itau - nt;

    float tau = lag_index * dt / 2.0f;

    P[itau] = (fabsf(tau) <= tau0) ? tau : 0.0f;
  }

  return P;
}

float* get_c_1d(const float* u_s, const float* u_o, int nt)
{
  float complex* U_s = get_fft_1d((float*)u_s, nt);

  float complex* U_o = get_fft_1d((float*)u_o, nt);

  float complex* cross = malloc(nt * sizeof(*cross));

  for (int iw = 0; iw < nt; ++iw)
  {
    cross[iw] = conjf(U_o[iw]) * U_s[iw];
  }

  float* c = get_ifft_1d(cross, nt);

  free(U_s);
  free(U_o);
  free(cross);

  return c;
}

float get_cross_1d(
    float* u_s,
    float* u_o,
    float dt,
    int nt,
    float tau0
)
{
  float* P = get_penalty(tau0, nt, dt);

  float* c = get_c_1d(u_s, u_o, nt);

  double result = 0.0;

  if(!initialized)
  {
    //plot1d(P, nt);
    initialized = 1;
  }

  for (int itau = 0; itau < nt; ++itau)
  {
    double pc = (double)P[itau] * (double)c[itau];

    result += pc * pc;
  }

  free(P);
  free(c);

  return (float)(0.5 * result);
}

#include <complex.h>
#include <math.h>
#include <stdlib.h>

#include "fft.h"
#include "plot.h"

#define M_PI 3.14159265f

static int initialized = 0;

int get_dtau(_Complex float* U_s, _Complex float* U_o, int nt)
{
  float complex* cross = malloc(nt * sizeof(*cross));

  for (int iw = 0; iw < nt; ++iw)
    cross[iw] = U_o[iw] * conjf(U_s[iw]);

  float* c0 = get_ifft_1d(cross, nt);

  float max = -INFINITY;
  int dtau = 0;

  for (int it = 0; it < nt; ++it)
  {
    if (c0[it] > max)
    {
      max = c0[it];
      dtau = it;
    }
  }

  if (dtau > nt / 2) dtau -= nt;

  free(cross); free(c0);

  return dtau;
}

float get_c_zhang_1d(
  const float* u_s,
  const float* u_o,
  int dtau,
  int nt
)
{
  float numerator = 0.0f;
  float energy_o = 0.0f;
  float energy_s = 0.0f;

  for (int it = 0; it < nt; ++it)
  {
    int io = it + dtau;

    if (io < 0 || io >= nt)
      continue;

    float d = u_o[io];
    float u = u_s[it];

    numerator += d * u;

    energy_o += d * d;
    energy_s += u * u;
  }

  float denominator =
    sqrtf(energy_o * energy_s);

  if (denominator == 0.0f)
    return 0.0f;

  return numerator / denominator;
}

float get_cross_zhang_1d(
  const float* u_s,
  const float* u_o,
  float dt,
  int nt,
  float tau0
)
{
  float complex* U_s =
    get_fft_1d((float*)u_s, nt);

  float complex* U_o =
    get_fft_1d((float*)u_o, nt);

  int dtau =
    get_dtau(U_s, U_o, nt);

  float c =
    get_c_zhang_1d(
      u_s,
      u_o,
      dtau,
      nt
    );

  float delta_tau =
    dtau * dt;

  float result =
    c * delta_tau * delta_tau;

  free(U_s);
  free(U_o);

  return result;
}


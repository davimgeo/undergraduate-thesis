#include <complex>

//#include "2D/objf.h"
#include "2D/dft2d.h"
#include "utils.h"

#include "../include/plot.h"

typedef std::complex<float> complex;

float* get_c(
  float* u_s, float* u_o,
  float dt, float dk,
  int nt, int nrec
)
{
  DFTOperator T = dft_operator_2d(dt, dk, nt, nrec);
  IDFTOperator TI = idft_operator_objf_2d(dt, dk, nt, nrec);

  complex* conj_im_u_s = conjugate2d(
    computeDFT(nt, nrec, u_s, T), nt, nrec
  );
  complex* im_u_o = computeDFT(nt, nrec, u_o, T);

  complex* correlation = new complex[nt * nrec];

  for (int i = 0; i < nt; ++i) {
    for (int j = 0; j < nrec; ++j) {
      int idx = i * nrec + j;

      correlation[idx] = conj_im_u_s[idx] * im_u_o[idx];
    }
  }

  return computeIDFT(correlation, TI, nt, nrec);
}

float* get_penalty(int nt, float t0)
{
  float* P = new float[nt];

  for (int i = 0; i < nt; ++i) {
    float tau = (i - (float)nt/2);

    if(std::abs(tau) <= t0) {
      P[i] = tau;
    } else {
      P[i] = 0.0f;
    }
  }  

  return P;
}

float get_correlation_objf(
  float *u_s, float *u_o,
  float dt, float dk,
  int nt, int nrec,
  float t0
)
{
  float result = 0.0f;

  float* c = get_c(u_s, u_o, dt, dk, nt, nrec);
  float* P = get_penalty(nt, t0);

  for (int tau = 0; tau < nt; ++tau) {

      for (int j = 0; j < nrec; ++j) {

        int idx = tau * nrec + j;

        float pc = P[tau] * c[idx];

        result += pc * pc;
      }
  }

  delete[] c;
  delete[] P;

  return 0.5f * result;
}


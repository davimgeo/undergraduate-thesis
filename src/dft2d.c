#include <cmath>
#include <complex>

#include "utils.h"
#include "2D/dft2d.h"

DFTOperator dft_operator_2d(float dt, float dh, int M, int N)
{
  struct DFTOperator d;

  d.Tx = new std::complex<float>[N*N];
  d.Tz = new std::complex<float>[M*M];

  float df = 1.0f / (M*dt);
  float dk = 1.0f / (N*dh);

  #pragma omp parallel for schedule(static)
  for (int f = 0; f < M; f++) {
    for (int n = 0; n < M; n++) {
      float angle = -2.0f * M_PI * f*df * n*dt;

      d.Tz[f * M + n] =
        std::exp(std::complex<float>(0.0f, angle));
    }
  }

  #pragma omp parallel for schedule(static)
  for (int x = 0; x < N; x++) {
    for (int n = 0; n < N; n++) {
      float angle = -2.0f * M_PI * x*dk * n*dh;

      d.Tx[x * N + n] = 
        std::exp(std::complex<float>(0.0f, angle));
    }
  }

  return d;
}

IDFTOperator idft_operator_objf_2d(float dt, float dh, int M, int N)
{
  struct IDFTOperator d;

  d.C_Tx = new std::complex<float>[N*N];
  d.C_Tz_T = new std::complex<float>[M*M];

  float df = 1.0f / (M*dt);
  float dk = 1.0f / (N*dh);

  #pragma omp parallel for schedule(static)
  for (int f = 0; f < M; f++) {
    for (int n = 0; n < M; n++) {
      float w = 2.0f * M_PI * f*df;
      float tau = (n - (float)M/2) * dt;
      // 2iwtau
      float angle = 2.0f * (w * tau);

      d.C_Tz_T[n * M + f] =
        std::exp(std::complex<float>(0.0f, angle));
    }
  }

  #pragma omp parallel for schedule(static)
  for (int x = 0; x < N; x++) {
    for (int n = 0; n < N; n++) {
      float w = 2.0f * M_PI * x*dk;
      // 2iwk
      float angle = 2.0f * (w * n*dh);

      d.C_Tx[x * N + n] = 
        std::exp(std::complex<float>(0.0f, angle));
    }
  }

  return d;
}

IDFTOperator idft_operator_2d(float dt, float dh, int M, int N)
{
  struct IDFTOperator d;

  d.C_Tx = new std::complex<float>[N*N];
  d.C_Tz_T = new std::complex<float>[M*M];

  float df = 1.0f / (M*dt);
  float dk = 1.0f / (N*dh);

  #pragma omp parallel for schedule(static)
  for (int f = 0; f < M; f++) {
    for (int n = 0; n < M; n++) {
      float w = +2.0f * M_PI * f*df;
      float angle = w * n*dt;

      d.C_Tz_T[n * M + f] =
        std::exp(std::complex<float>(0.0f, angle));
    }
  }

  #pragma omp parallel for schedule(static)
  for (int x = 0; x < N; x++) {
    for (int n = 0; n < N; n++) {
      float w = 2.0f * M_PI * x*dk;
      float angle = w * n*dh;

      d.C_Tx[x * N + n] = 
        std::exp(std::complex<float>(0.0f, angle));
    }
  }

  return d;
}

std::complex<float>* computeDFT(int M, int N, float* f, const DFTOperator& d)
{
  // DFT = Tz * f * Tx^T
  
  std::complex<float>* Tzf =
    mat_mult<std::complex<float>>(d.Tz, f, M, M, M, N);

  std::complex<float>* TxT =
    transpose(d.Tx, N, N);

  std::complex<float>* DFT =
    mat_mult<std::complex<float>>(Tzf, TxT, M, N, N, N);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      DFT[i * N + j] /= M*N;
    }
  }

  delete[] Tzf;
  delete[] TxT;

  return DFT;
}

float* computeIDFT(std::complex<float>* F, const IDFTOperator& id, int M, int N)
{
  // f = conj(Tz^T) * F * conj(Tx)

  std::complex<float>* conj_TzT_F = 
    mat_mult<std::complex<float>>(id.C_Tz_T, F, M, M, M, N);

  std::complex<float>* conj_TzT_F_conj_Tx =
    mat_mult<std::complex<float>>(
      conj_TzT_F, id.C_Tx, M, N, N, N
    );

  auto* IDFT = new float[M * N];

  for (int i = 0; i < M * N; ++i) {
    IDFT[i] = conj_TzT_F_conj_Tx[i].real();
  }

  return IDFT;
}



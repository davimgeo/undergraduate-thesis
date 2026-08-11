#include <stdlib.h>

#include "utils.h"

#include "1D/ricker.h"
#include "1D/euclidian.h"
#include "1D/cross.objf.h"
#include "1D/decon_objf.h"

float* moving_rickers_l1(
  float* wavelet,
  int nt,
  int result_size
)
{
  float* result = (float*)malloc(result_size * sizeof(float));

  float phase = 0.0f;

  for (int i = 0; i < result_size; i++)
  {
    float* ricker_phase = get_ricker(
      nt,
      30.0f,
      1e-3f,
      phase
    );

    result[i] = l1_norm_1d(
      wavelet,
      ricker_phase,
      nt
    );

    phase += 0.7f / result_size;

    free(ricker_phase);
  }

  normalize(result, result_size);

  return result;
}

float* moving_rickers_l2(
  float* wavelet,
  int nt,
  int result_size
)
{
  float* result = (float*)malloc(result_size * sizeof(float));

  float phase = 0.0f;

  for (int i = 0; i < result_size; i++)
  {
    float* ricker_phase = get_ricker(
      nt,
      30.0f,
      1e-3f,
      phase
    );

    result[i] = l2_norm_1d(
      wavelet,
      ricker_phase,
      nt
    );

    phase += 0.7f / result_size;

    free(ricker_phase);
  }

  normalize(result, result_size);

  return result;
}

float* moving_rickers_cross(
  float* u_o,
  int nt,
  int result_size,
  float fmax,
  float dt,
  float t0
)
{
  float* result = (float*)malloc(result_size * sizeof(float));

  float phase = 0.0f;

  for (int i = 0; i < result_size; i++)
  {
    float* u_s = get_ricker(
      nt,
      fmax,
      dt,
      phase
    );

    result[i] = get_cross_result(
      u_s,
      u_o,
      dt,
      nt,
      t0
    );

    phase += 0.7f / result_size;

    free(u_s);
  }

  normalize(result, result_size);

  return result;
}

float* moving_rickers_decon(
  float* u_o,
  int nt,
  int result_size,
  float fmax,
  float dt,
  float t0
)
{
  float* result = (float*)malloc(result_size * sizeof(float));

  float phase = 0.0f;

  for (int i = 0; i < result_size; i++)
  {
    float* u_s = get_ricker(
      nt,
      fmax,
      dt,
      phase
    );

    result[i] = get_decon_result(
      u_s,
      u_o,
      dt,
      nt,
      t0
    );

    phase += 0.7f / result_size;

    free(u_s);
  }

  normalize(result, result_size);

  return result;
}


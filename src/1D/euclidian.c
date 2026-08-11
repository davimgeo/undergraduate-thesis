#include <math.h>
#include <stdlib.h>

#include "utils.h"
#include "1D/ricker.h"

float l1_norm_1d(float* A, float* B, int size)
{
  float result = 0.0f;
  for (int i = 0; i < size; i++) 
  {
    result += fabsf(A[i] - B[i]);
  }

  return result;
}

float l1_norm_2d(float* A, float* B, int row, int col)
{
  float result = 0.0f;
  for (int i = 0; i < row; i++) 
  {
    for (int j = 0; j < col; j++) 
    {
      int idx = i * col + j;
      result += fabsf(A[idx] - B[idx]);
    }
  }

  return result;
}

float l2_norm_1d(float* A, float* B, int size)
{
  float result = 0.0f;
  for (int i = 0; i < size; i++) 
  {
    result += (A[i] - B[i]) * (A[i] - B[i]);
  }

  return sqrtf(result);
}

float l2_norm_2d(float* A, float* B, int row, int col)
{
  float result = 0.0f;
  for (int i = 0; i < row; i++) 
  {
    for (int j = 0; j < col; j++) 
    {
      int idx = i * col + j;
      result += (A[idx] - B[idx]) * (A[idx] - B[idx]);
    }
  }
  return sqrtf(result);
}


float* get_l1_result(
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


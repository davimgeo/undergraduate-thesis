#include <stdlib.h>
#include <stdbool.h>

#include "IO.h"
#include "utils.h"

#include "ricker.h"
#include "moving_rickers.h"

#include "plot.h"

int main()
{
  PROFILE_BEGIN();

  int nt = 1001;
  int result_size = 200;

  float fmax = 30.0f;
  float dt = 1e-3f;

  float t0 = 0.3f;

  #if 1
  float* ricker1 = get_ricker(nt, fmax, dt, 0.6f);
  float* ricker2 = get_ricker(nt, fmax, dt, 0.2f);

  #else
 
  float* ricker = get_ricker(nt, fmax, dt, 0.25f);

  #endif

  float* cross = moving_rickers_cross(ricker1, nt, result_size, fmax, dt, t0);
  float* l2 = moving_rickers_l2(ricker1, nt, result_size, fmax, dt);
  float* decon = moving_rickers_decon(ricker1, nt, result_size, fmax, dt, t0);

  float* cross2 = moving_rickers_cross(ricker2, nt, result_size, fmax, dt, t0);
  float* l22 = moving_rickers_l2(ricker2, nt, result_size, fmax, dt);
  float* decon2 = moving_rickers_decon(ricker2, nt, result_size, fmax, dt, t0);

  float* cross_result = (float*)malloc(result_size * sizeof(float));
  float* l2_result = (float*)malloc(result_size * sizeof(float));
  float* decon_result = (float*)malloc(result_size * sizeof(float));

  for (int i = 0; i < result_size; i++) 
  {
    cross_result[i] = cross[i] + cross2[i];
    decon_result[i] = decon[i] + decon2[i];
    l2_result[i] = l2[i] + l22[i];
  }

  normalize(cross_result, result_size); normalize(l2_result, result_size);
  normalize(cross, result_size); normalize(cross2, result_size);

  plot1d_compare(cross, cross2, result_size);
  plot1d_compare(cross_result, l2_result, result_size);

  PROFILE_END();
}







#include <math.h>
#include <stdlib.h>

#define PI 3.14159265358979323846f

float* get_ricker(
  int nt,
  float fmax,
  float dt,
  float phase
)
{
  float* wavelet = (float*)malloc(sizeof(float)*nt);

  float t0 = 2.0f * PI / fmax;

  float fc = fmax / (3.0f*sqrtf(PI));

  for (int i = 0; i < nt; i++) 
  {
    float t = (i * dt) - t0 - phase; 
    float arg = PI * (PI * PI * fc * fc * t * t);
    wavelet[i] = (1.0f - 2.0f * arg) * expf(-arg);
  }

  return wavelet;
}

#include "plot.h"
#include "utils.h"
#include <math.h>

#define SIZE 201

#define XMIN -2.0f
#define XMAX 2.0f

#define YMIN -1.0f
#define YMAX 3.0f

#define PI 3.14159f
#define E  2.71828f

int main()
{
  PROFILE_BEGIN();

  float* rosenbrock = (float*)malloc(SIZE * SIZE * sizeof(float));
  float* ackley     = (float*)malloc(SIZE * SIZE * sizeof(float));
  float* x_arr      = (float*)malloc(SIZE * sizeof(float));
  float* y_arr      = (float*)malloc(SIZE * sizeof(float));

  float dx = (XMAX - XMIN) / (float)(SIZE - 1);
  float dy = (XMAX - XMIN) / (float)(SIZE - 1);

  for (int ix = 0; ix < SIZE; ix++) 
  {
    x_arr[ix] = XMIN + ix*dx;
    float x = x_arr[ix];

    for (int iy = 0; iy < SIZE; iy++) 
    {
      y_arr[iy] = YMIN + iy*dy;
      float y = y_arr[iy];

      int idx = ix * SIZE + iy;
      rosenbrock[idx] = (1.0f - x)*(1.0f - x) + 10.0f*(y - x*x)*(y - x*x);

      ackley[idx] = -20.0f * expf(-0.2f * sqrtf(0.5f * x*x + y*y)) -
                    expf(0.5f * (cosf(2.0f * PI * x)) + cosf(2.0f * PI * y)) +
                    E + 20.0f;
    }     
  } 

  //plot3d(rosenbrock, x_arr, y_arr, SIZE, SIZE);
  //plot3d(ackley, x_arr, y_arr, SIZE, SIZE);

  PROFILE_END();

  return 0;
}







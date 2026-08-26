#include "plot.h"
#include "utils.h"
#include <math.h>

#define SIZE 201

#define XMIN -2.0f
#define XMAX 2.0f

#define YMIN -2.0f
#define YMAX 2.0f

#define PI 3.14159f
#define E  2.71828f

#define TOL 1e-9
#define MAX_ITERATIONS 20

typedef struct Point 
{
  float x; 
  float y;
} Point; 

float* get_rosenbrock(void)
{
  float* rosenbrock = malloc(SIZE * SIZE * sizeof(*rosenbrock));

  float dx = (XMAX - XMIN) / (float)(SIZE - 1);
  float dy = (YMAX - YMIN) / (float)(SIZE - 1);

  for (int ix = 0; ix < SIZE; ix++)
  {
    float x = XMIN + ix*dx;

    for (int iy = 0; iy < SIZE; iy++)
    {
      float y = YMIN + iy*dy;

      int idx = ix * SIZE + iy;

      rosenbrock[idx] =
        (1.0f - y)*(1.0f - y) +
        10.0f*(x - y*y)*(x - y*y);
    }
  }

  return rosenbrock;
}

float get_rosenbrock_value(Point p)
{
 return (1.0f - p.y)*(1.0f - p.y) + 
        10.0f*(p.x - p.y*p.y)*(p.x - p.y*p.y);
}

Point get_rosenbrock_gradient(Point p)
{
  Point nabla;

  nabla.x = 40.0f * p.x*p.x*p.x - 40.0f*p.x*p.y + 2.0f*p.x - 2.0f;
  nabla.y = 20.0f * (p.y - p.x*p.x);

  return nabla;
}

float l2_norm(float dobs, float dcalc)
{
  return 0.5f * (dobs - dcalc) * (dobs - dcalc);
}

int main()
{
  PROFILE_BEGIN();

  float* rosenbrock = get_rosenbrock();

  contourplot(rosenbrock, SIZE, SIZE, XMIN, XMAX, YMIN, YMAX);

  /**************************** Steepest Descent ******************************/

  Point mreal = {.x = 1.0f, .y = 1.0f}; // global minimum
  float dobs = get_rosenbrock_value(mreal);

  Point m_current  = {.x = 0.3f, .y = 0.7f};
  float a_k = 0.01f;

  for (int it = 0; it < MAX_ITERATIONS; it++) 
  {
    Point mk = m_current;
    float dcalc_0 = get_rosenbrock_value(mk);
    float chi_mk = l2_norm(dobs, dcalc_0);
    Point nabla_chi = get_rosenbrock_gradient(mk);

    // m_{k+1} = m_k - a_k\nabla\chi(m_k)
    Point mk1 = {
      .x = mk.x - a_k*nabla_chi.x, 
      .y = mk.y - a_k*nabla_chi.y
    };
    float dcalc_1 = get_rosenbrock_value(mk1); 
    float chi_mk1 = l2_norm(dobs, dcalc_1);

    if(chi_mk1 < chi_mk)
    {
      // update alpha_k
      // m_current = m_k+1
    }

    if((chi_mk / dobs) < TOL)
      goto END;
      
  }

  END:
    PROFILE_END();

    free(rosenbrock);

  return 0;
}






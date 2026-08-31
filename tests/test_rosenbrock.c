#include "plot.h"
#include "utils.h"
#include <math.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define SIZE 501

#define XMIN -2.0f
#define XMAX 2.0f
#define YMIN -2.0f
#define YMAX 2.0f

#define PI 3.14159f
#define E  2.71828f

#define TOL 1e-6
#define MAX_ITERATIONS 2001

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
        (1.0f - x)*(1.0f - x) +
        10.0f*(y - x*x)*(y - x*x);
    }
  }

  return rosenbrock;
}

float get_rosenbrock_value(Point p)
{
 return (1.0f - p.x)*(1.0f - p.x) + 
        10.0f*(p.y - p.x*p.x)*(p.y - p.x*p.x);
}

Point get_nabla_gradient(Point p, float dcalc, float dobs)
{
  Point nabla;

  float r = (dcalc - dobs);

  nabla.x = (40.0f * p.x*p.x*p.x - 40.0f*p.x*p.y + 2.0f*p.x - 2.0f) * r;
  nabla.y = (20.0f * (p.y - p.x*p.x)) * r;

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

  Point m_current  = {.x = -0.5f, .y = 0.5f};
  float a_k = 0.01f * MAX(m_current.x, m_current.y);

  int i = 0; 
  for (int it = 0; it < MAX_ITERATIONS; it++) 
  {
    /* mk = m_current
     * dcalc = G(m_k)
     * \chi(m_k) = ||dobs - dcalc||^2_2
     * \nabla\chi(m_k) = \Delta d \nabla f */
    Point mk = m_current;
    float dcalc_0 = get_rosenbrock_value(mk);
    float chi_mk = l2_norm(dobs, dcalc_0);
    Point nabla_chi = get_nabla_gradient(mk, dcalc_0, dobs);

    /* m_{k+1} = m_k - a_k\nabla\chi(m_k)
     * dcalc = G(m_{k+1})
     * \chi(m_{k+1}) = ||d_bs - dcalc||^2_2 */
    Point mk1 = {
      .x = mk.x - a_k*nabla_chi.x, 
      .y = mk.y - a_k*nabla_chi.y
    };
    float dcalc_1 = get_rosenbrock_value(mk1); 
    float chi_mk1 = l2_norm(dobs, dcalc_1);
   
    /*\chi(m_{k+1}) <= \chi(m_k) - c_1 \alpha_k ||\nable\chi(m_k)||^2_2 */
    float grad_norm2 = nabla_chi.x*nabla_chi.x + nabla_chi.y*nabla_chi.y;
    float armijo_condition = chi_mk - 1e-4f*a_k*grad_norm2;

    if(chi_mk1 < armijo_condition)
    {
      m_current = mk1;
      a_k = 0.01f * MAX(m_current.x, m_current.y);
      if(!(i % 10)) printf("m_current: (%g, %g)\n", m_current.x, m_current.y);
      //if(!(i % 10))printf("a_k: %g\n", a_k);
      i++;
    }

    //if((chi_mk / dobs) < TOL)
      //goto END;
      
  }

  END:
    PROFILE_END();

    free(rosenbrock);

  return 0;
}







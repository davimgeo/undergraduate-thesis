#include "plot.h"
#include "utils.h"
#include <math.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define SIZE 501

#define XMIN -2.0f
#define XMAX 2.0f
#define YMIN -2.0f
#define YMAX 2.0f

#define PI 3.14159f

#define TOL 1e-8f
#define MAX_ITERATIONS 1001

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

 float norm = sqrtf(
    nabla.x*nabla.x +
    nabla.y*nabla.y
  );

  if (norm > 0.0f)
  {
    nabla.x /= norm;
    nabla.y /= norm;
  }
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

  /**************************** Steepest Descent ******************************/

  Point mreal = {.x = 1.0f, .y = 1.0f}; // global minimum
  float dobs = get_rosenbrock_value(mreal);

  Point m_current = {.x = -0.5f, .y = 0.5f};

  float dcalc_0 = get_rosenbrock_value(m_current);
  float chi_m0 = l2_norm(dobs, dcalc_0);

  float a_k = 1.0f;

  float* model_updates_x =
    (float*)malloc((MAX_ITERATIONS + 1) * sizeof(float));

  float* model_updates_y =
    (float*)malloc((MAX_ITERATIONS + 1) * sizeof(float));

  // Store initial model m0
  int final_model_idx = 0;

  model_updates_x[final_model_idx] = m_current.x;
  model_updates_y[final_model_idx] = m_current.y;
  final_model_idx++;

  for (int it = 0; it < MAX_ITERATIONS; it++) 
  {
    // mk = m_current
    Point mk = m_current;

    // dcalc = G(m_k)
    float dcalc_current = get_rosenbrock_value(mk);

    // chi(m_k)
    float chi_mk = l2_norm(dobs, dcalc_current);

    // nabla chi(m_k)
    Point nabla_chi =
      get_nabla_gradient(mk, dcalc_current, dobs);

    // m_{k+1} = m_k - a_k nabla chi(m_k)
    Point mk1 = {
      .x = mk.x - a_k*nabla_chi.x, 
      .y = mk.y - a_k*nabla_chi.y
    };

    // dcalc = G(m_{k+1})
    float dcalc_1 = get_rosenbrock_value(mk1); 

    // chi(m_{k+1})
    float chi_mk1 = l2_norm(dobs, dcalc_1);

    // Armijo condition
    float grad_norm2 =
      nabla_chi.x*nabla_chi.x +
      nabla_chi.y*nabla_chi.y;

    float armijo_condition =
      chi_mk - 1e-4f*a_k*grad_norm2;

    if (chi_mk1 < armijo_condition)
    {
      m_current = mk1;

      model_updates_x[final_model_idx] = m_current.x;
      model_updates_y[final_model_idx] = m_current.y;
      final_model_idx++;
    } else {
      a_k *= 0.5f;
    }

    if ((chi_mk / chi_m0) <= TOL)
    {
      printf("Last Iteration: %d\n", it);
      goto END;
    }
  }

END:

  PROFILE_END();

  printf(
    "m_current: (%g, %g)\n",
    m_current.x,
    m_current.y
  );

  contourplot_opt(
    rosenbrock,
    model_updates_x,
    model_updates_y,
    final_model_idx,
    SIZE,
    SIZE,
    XMIN,
    XMAX,
    YMIN,
    YMAX
  );

  free(model_updates_y);
  free(model_updates_x);
  free(rosenbrock);

  return 0;
}

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

#define TOL 1e-8f
#define MAX_ITERATIONS 10
#define C1 1e-4f

typedef struct Point 
{
  float x; 
  float y;
} Point; 

float* get_ackley(void)
{
  float* ackley = malloc(SIZE * SIZE * sizeof(*ackley));

  float dx = (XMAX - XMIN) / (float)(SIZE - 1);
  float dy = (YMAX - YMIN) / (float)(SIZE - 1);

  for (int ix = 0; ix < SIZE; ix++)
  {
    float x = XMIN + ix*dx;

    for (int iy = 0; iy < SIZE; iy++)
    {
      float y = YMIN + iy*dy;

      int idx = ix * SIZE + iy;
      ackley[idx] =
        -20.0f * expf(-0.2f * sqrtf(0.5f * (x*x + y*y))) -
        expf(0.5f * (cosf(2.0f * PI * x) + cosf(2.0f * PI * y))) +
        E + 20.0f;
    }
  }

  return ackley;
}

float get_ackley_value(Point p)
{
  return
    -20.0f * expf(-0.2f * sqrtf(0.5f * (p.x*p.x + p.y*p.y))) -
    expf(0.5f * (cosf(2.0f * PI * p.x) + cosf(2.0f * PI * p.y))) +
    E + 20.0f;
}

Point get_nabla_gradient(Point p, float dcalc, float dobs)
{
  Point nabla;

  float r = dcalc - dobs;

  float radius = sqrtf(0.5f * (p.x*p.x + p.y*p.y));

  if (radius == 0.0f)
  {
    nabla.x = 0.0f;
    nabla.y = 0.0f;
    return nabla;
  }

  nabla.x =
    (2.0f * p.x / radius) *
    expf(-0.2f * radius) +
    PI * sinf(2.0f * PI * p.x) *
    expf(0.5f * (
      cosf(2.0f * PI * p.x) +
      cosf(2.0f * PI * p.y)
    ));

  nabla.y =
    (2.0f * p.y / radius) *
    expf(-0.2f * radius) +
    PI * sinf(2.0f * PI * p.y) *
    expf(0.5f * (
      cosf(2.0f * PI * p.x) +
      cosf(2.0f * PI * p.y)
    ));

  nabla.x *= r;
  nabla.y *= r;

  return nabla;
}

float l2_norm(float dobs, float dcalc)
{
  return 0.5f * (dobs - dcalc) * (dobs - dcalc);
}

int main()
{
  PROFILE_BEGIN();

  float* ackley = get_ackley();

  /**************************** Steepest Descent ******************************/

  Point mreal = {.x = 0.0f, .y = 0.0f}; // global minimum
  float dobs = get_ackley_value(mreal);

  Point m_current = {.x = -1.5f, .y = 0.5f};

  float dcalc_0 = get_ackley_value(m_current);
  float chi_m0 = l2_norm(dobs, dcalc_0);

  float* model_updates_x = (float*)malloc((MAX_ITERATIONS + 1) * sizeof(float));
  float* model_updates_y = (float*)malloc((MAX_ITERATIONS + 1) * sizeof(float));

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
    float dcalc_current = get_ackley_value(mk);
    // chi(m_k)
    float chi_mk = l2_norm(dobs, dcalc_current);
    // nabla chi(m_k)
    Point nabla_chi = get_nabla_gradient(mk, dcalc_current, dobs);

    float grad_norm = sqrtf(
      nabla_chi.x*nabla_chi.x +
      nabla_chi.y*nabla_chi.y
    );
    if (grad_norm <= TOL) goto END;

    // normalized descent direction
    Point h_k = {
      .x = -nabla_chi.x / grad_norm,
      .y = -nabla_chi.y / grad_norm
    };
    float gTp = nabla_chi.x*h_k.x + nabla_chi.y*h_k.y;

    float a_k = 1.0f;

    // line search
    for (int i = 0; i < 30; i++)
    {
      // m_{k+1} = m_k - a_k \nabla\chi(m_k)
      Point mk1 = {
        .x = mk.x + a_k*h_k.x,
        .y = mk.y + a_k*h_k.y
      };
      // dcalc = G(m_{k+1})
      float dcalc_1 = get_ackley_value(mk1);
      // chi(m_{k+1})
      float chi_mk1 = l2_norm(dobs, dcalc_1);

      int armijo = chi_mk1 <= chi_mk + C1*a_k*gTp;
      if (armijo)
      {
        m_current = mk1;

        model_updates_x[final_model_idx] = m_current.x;
        model_updates_y[final_model_idx] = m_current.y;
        final_model_idx++;

        break;
      }

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

    printf("m_current: (%g, %g)\n", m_current.x, m_current.y);

    contourplot_opt(
      ackley, model_updates_x, model_updates_y, final_model_idx,
      SIZE, SIZE, XMIN, XMAX, YMIN, YMAX
    );

    free(model_updates_y);
    free(model_updates_x);
    free(ackley);

  return 0;
}











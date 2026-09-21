#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "IO.h"
#include "propagation.h"
#include "utils.h"
#include "rtm.h"
#include "config/config.h"
#include "plot.h"

#define NT 4001
#define NREC 114
#define NSHOT 43

#define TOL 1e-8f
#define C1 1e-4f

#define MAX_ITERATIONS 20
#define MAX_LINE_SEARCH 10

float* get_nabla_gradient(float* vp, const float* dobs, SpecsContext* specs)
{
  size_t model_size = (size_t)specs->model.nz * specs->model.nx;

  geometry_t* geom = Geometry_InitCreate(NULL, &specs->geometry);
  Geometry_Create(geom, 0);
  //printf("%d\n", geom->nsrc);

  wavelet_t* wave = Wavelet_Init(NULL, &specs->wavelet);
  Wavelet_Create(wave);
  Wavelet_SecondDerivative(wave);

  model_t* model = Model_Init(NULL, &specs->model);
  Model_Set(model, vp);
  Model_Extent(model);

  seismogram_t* seis = Seismogram_Init(NULL, &specs->seismogram, geom->nrec, 0);

  propagation_t* prop = Propagation_Init(
    NULL,
    &specs->propagation,
    model, geom, wave, seis,
    PROPAGATION_ACOUSTIC
  );

  rtm_t* rtm = RTM_Init(NULL, prop);
  RTMv2_Run(rtm, dobs);

  float* nabla_chi = malloc(model_size * sizeof(float));

  for (int i = 0; i < model->nz; ++i)
  {
    for (int j = 0; j < model->nx; ++j)
    {
      size_t idx_im = (size_t)(i + model->nb) * model->nxx + (j + model->nb);
      size_t idx_nabla = (size_t)i * model->nx + j;

      float v = vp[idx_nabla];

      nabla_chi[idx_nabla] = rtm->image[idx_im];
    }
  }

  RTM_Destroy(rtm);
  Propagation_Destroy(prop);

  Seismogram_Destroy(seis);
  Model_Destroy(model);
  Wavelet_Destroy(wave);
  Geometry_Destroy(geom);

  return nabla_chi;
}

float* get_dcalc(float* vp, SpecsContext* specs)
{
  printf("Started Dcalc\n");

  size_t shot_size = (size_t)NT * NREC;
  size_t data_size = (size_t)NSHOT * shot_size;

  float* dcalc = malloc(data_size * sizeof(float));

  wavelet_t* wave = Wavelet_Init(NULL, &specs->wavelet);
  Wavelet_Create(wave);

  model_t* model = Model_Init(NULL, &specs->model);
  Model_Set(model, vp);
  Model_Extent(model);

  for (int ishot = 0; ishot < NSHOT; ++ishot)
  {
    geometry_t* geom = Geometry_InitCreate(NULL, &specs->geometry);
    Geometry_Create(geom, GEOMETRY_ONLY_RECEIVERS);
    Geometry_SetSource(
      geom,
      ishot * specs->geometry.offset_src,
      specs->geometry.src_depth
    );

    seismogram_t* seis = Seismogram_Init(NULL, &specs->seismogram, geom->nrec, 0);

    propagation_t* prop = Propagation_Init(
      NULL,
      &specs->propagation,
      model, geom, wave, seis,
      PROPAGATION_ACOUSTIC
    );

    Propagation_Run(prop, 0);

    memcpy(
      dcalc + (size_t)ishot * shot_size,
      seis->seismogram,
      shot_size * sizeof(float)
    );

    Propagation_Destroy(prop);
    Seismogram_Destroy(seis);
    Geometry_Destroy(geom);
  }

  Model_Destroy(model);
  Wavelet_Destroy(wave);

  printf("Finished Dcalc\n");
  return dcalc;
}

float l2_norm(const float* dcalc, const float* dobs, int nt, int nrec, int nshot)
{
  float result = 0.0;

  for(int ishot = 0; ishot < nshot; ++ishot)
  {
    // get specific shot from dobs/dcalc
    const float* u_s = dcalc + ishot * nt * nrec;
    const float* u_o = dobs + ishot * nt * nrec;
    //plot2d(u_o, nt, nrec);

    for (int irec = 0; irec < nrec; ++irec)
    {
      for (int t = 0; t < nt; ++t)
      {
        int idx = t * nrec + irec;

        float r = u_s[idx] - u_o[idx];
        //printf("%g\n", u_o[idx]);

        result += r * r;
      }
    }
  }

  return 0.5f * result;
}

float gradient_norm(const float* gradient, size_t size)
{
  float result = 0.0;

  for (size_t i = 0; i < size; ++i)
  {
    float g = gradient[i];
    result += g * g;
  }

  return sqrtf(result);
}

void save_current(float* m_current, SpecsContext* specs, int it)
{
  char filename[256];

  snprintf(
    filename,
    sizeof(filename),
    "data/FWI/m_%01d.bin",
    it + 1
  );

  write2d(filename, m_current, sizeof(float), 351, 881);
}

float get_GTP(const float* nabla_chi, size_t size)
{
  float gTp = 0.0;

  for (size_t i = 0; i < size; i++)
  {
    float h_k = -nabla_chi[i];

    gTp += nabla_chi[i] * h_k;
  }

  return gTp;
}

void get_slowness_from_velocity(
  const float* velocity,
  float* slowness,
  int nz,
  int nx
)
{
  for(int i = 0; i < nz*nx; i++)
    slowness[i] = 1.0f / (velocity[i]*velocity[i]);
}

void get_velocity_from_slowness(
  const float* slowness,
  float* velocity,
  int nz,
  int nx
)
{
  for (int i = 0; i < nz * nx; i++)
    velocity[i] = 1.0f / sqrtf(slowness[i]);
}

float get_initial_alpha(float* model, int row, int col)
{
  float max = model[0];
  float min = model[0];

  for(int i = 0; i < row * col; i++)
  {
    if(model[i] > max)
      max = model[i];

    if(model[i] < min)
      min = model[i];
  }

  return 0.01f * (max - min);
}

int main()
{
  PROFILE_BEGIN();

  SpecsContext* specs = Specs_Init(NULL);

  size_t model_size = (size_t)specs->model.nx * specs->model.nz;

  float* m_real = read2d("data/FWI/marmousi_real_141x681x_dh25m.bin", 141, 681);
  //plot2d(m_real, 141, 681);

  float* dobs = read_any("data/FWI/dobs.bin", NT * NREC * NSHOT);

  float* m0 = read2d_fortran("data/FWI/m0.bin", 141, 681);
  float* dcalc_0 = read_any("data/FWI/dcalc_0.bin", NT * NREC * NSHOT);
  float chi_m0 = l2_norm(dcalc_0, dobs, NT, NREC, NSHOT);

  float* vp_current = malloc(model_size * sizeof(float));
  float* vp_k1 = malloc(model_size * sizeof(float));

  float* m_current = malloc(model_size * sizeof(float));
  float* mk1 = malloc(model_size * sizeof(float));

  memcpy(vp_current, m0, model_size * sizeof(float));
  get_slowness_from_velocity(vp_current, m_current, 141, 681);

  for (int it = 0; it < MAX_ITERATIONS; it++)
  {
    printf("\nIteration %d\n", it);

    // mk = m_current
    float* mk = m_current;

    // dcalc = G(m_k)
    float* dcalc_current = get_dcalc(vp_current, specs);

    // chi(m_k)
    float chi_mk = l2_norm(dcalc_current, dobs, NT, NREC, NSHOT);
    printf("chi_mk: %g\n", chi_mk);

    // nabla chi(m_k)
    float* nabla_chi = get_nabla_gradient(vp_current, dobs, specs);
    plot2d(nabla_chi, 141, 681);

    // normalized descent direction
    float grad_norm = gradient_norm(nabla_chi, model_size);
    for (size_t i = 0; i < model_size; i++) nabla_chi[i] /= grad_norm;
    float gTp = grad_norm * get_GTP(nabla_chi, model_size);

    float a_k = get_initial_alpha(mk, 141, 681);
    printf("alpha_0: %f\n", a_k);

    int accepted = 0;

    // line search
    for (int ils = 0; ils < MAX_LINE_SEARCH; ++ils)
    {
      for (size_t i = 0; i < model_size; ++i)
      {
        // m_{k+1} = m_k - a_k*nabla_chi(m_k)
        mk1[i] = mk[i] - a_k * nabla_chi[i];
        printf("%g\n", nabla_chi[i]);
      }

      get_velocity_from_slowness(mk1, vp_k1, 141, 681);

      // dcalc = G(m_{k+1})
      float* dcalc_1 = get_dcalc(vp_k1, specs);

      // chi(m_{k+1})
      float chi_mk1 = l2_norm(dcalc_1, dobs, NT, NREC, NSHOT);
      printf("chi_mk1: %g\n", chi_mk1);
      printf("armijo: %g\n", chi_mk + C1*a_k*gTp);

      int armijo = chi_mk1 <= chi_mk + C1*a_k*gTp;
      if (armijo)
      {
        printf("ACCEPTED\n");

        compare_diff(vp_current, vp_k1, 141, 681, "m_current", "mk1");

        float* temp = m_current;
        m_current = mk1;
        mk1 = temp;

        temp = vp_current;
        vp_current = vp_k1;
        vp_k1 = temp;

        accepted = 1;

        free(dcalc_1);

        break;
      }

      a_k *= 0.5f;

      free(dcalc_1);
    }

    free(nabla_chi);
    free(dcalc_current);

    if (!accepted)
    {
      printf("Line search failed at iteration %d\n", it);

      break;
    }

    save_current(vp_current, specs, it);
    //plot2d(m_current, 351, 881);

    if ((chi_mk / chi_m0) <= TOL)
    {
      printf("Converged at iteration %d.\n", it);

      break;
    }
  }

  free(vp_k1);
  free(vp_current);
  free(mk1);
  free(m_current);
  free(dcalc_0);
  free(m0);
  free(dobs);
  free(specs);

  PROFILE_END();

  return 0;
}

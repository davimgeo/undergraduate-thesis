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

#define TOL 1e-6f
#define C1 1e-4f

#define MAX_ITERATIONS 20
#define MAX_LINE_SEARCH 20
#define DIRECTIONAL_TEST_STEPS 4

typedef struct
{
  int nt;
  int nrec;
  int nshot;
} data_shape_t;

data_shape_t get_data_shape(SpecsContext* specs)
{
  geometry_t* geom = Geometry_InitCreate(NULL, &specs->geometry);
  Geometry_Create(geom, 0);

  data_shape_t shape;

  shape.nt = specs->seismogram.nt;
  shape.nrec = geom->nrec;
  shape.nshot = geom->nsrc;

  Geometry_Destroy(geom);

  return shape;
}

float* get_nabla_gradient(float* vp, SpecsContext* specs)
{
  size_t model_size = (size_t)specs->model.nz * specs->model.nx;

  geometry_t* geom = Geometry_InitCreate(NULL, &specs->geometry);
  Geometry_Create(geom, 0);

  wavelet_t* wave = Wavelet_Init(NULL, &specs->wavelet);
  Wavelet_Create(wave);

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
  RTMv2_Run(rtm, "data/FWI/dobs/");

  //plot_image(rtm, model, 10);

  float* nabla_chi = malloc(model_size * sizeof(float));

  for (int i = 0; i < model->nz; ++i)
  {
    for (int j = 0; j < model->nx; ++j)
    {
      size_t idx_im = (size_t)(i + model->nb) * model->nxx + (j + model->nb);
      size_t idx_nabla = (size_t)i * model->nx + j;

      if (i < 20)
        nabla_chi[idx_nabla] = 0.0f;
      else
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

float* get_dcalc(float* vp, SpecsContext* specs, data_shape_t shape)
{
  printf("Started Dcalc\n");

  size_t shot_size = (size_t)shape.nt * shape.nrec;
  size_t data_size = (size_t)shape.nshot * shot_size;

  float* dcalc = malloc(data_size * sizeof(float));

  wavelet_t* wave = Wavelet_Init(NULL, &specs->wavelet);
  Wavelet_Create(wave);

  model_t* model = Model_Init(NULL, &specs->model);
  Model_Set(model, vp);
  Model_Extent(model);

  for (int ishot = 0; ishot < shape.nshot; ++ishot)
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

double l2_norm(
  const float* dcalc,
  int nt,
  int nrec,
  int nshot,
  float dt
)
{
  double result = 0.0;

  size_t shot_size = (size_t)nt * nrec;

  for (int ishot = 0; ishot < nshot; ++ishot)
  {
    // get specific shot from dobs/dcalc
    const float* u_s = dcalc + (size_t)ishot * shot_size;

    char PATH[256];
    snprintf(
      PATH,
      sizeof(PATH),
      "%s/seismogram_%dx%d_shot%d.bin",
      "data/FWI/dobs/",
      nt,
      nrec,
      ishot
    );

    float* u_o = read2d(PATH, nt, nrec);
    //compare_diff(u_s, u_o, nt, nrec, "u_s", "u_o");

    for (int irec = 0; irec < nrec; ++irec)
    {
      for (int t = 0; t < nt; ++t)
      {
        size_t idx = (size_t)t * nrec + irec;

        double r = (double)u_s[idx] - (double)u_o[idx];

        result += r * r;
      }
    }

    free(u_o);
  }

  return 0.5 * result * (double)dt;
}

float gradient_scale(const float* gradient, size_t size)
{
  float scale = 0.0f;

  for (size_t i = 0; i < size; ++i)
  {
    float g = fabsf(gradient[i]);

    if (g > scale)
      scale = g;
  }

  return scale;
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

  write2d(
    filename,
    m_current,
    sizeof(float),
    specs->model.nz,
    specs->model.nx
  );
}

void get_slowness_from_velocity(
  const float* velocity,
  float* slowness,
  int nz,
  int nx
)
{
  for (int i = 0; i < nz * nx; i++)
    slowness[i] = 1.0f / (velocity[i] * velocity[i]);
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

  for (int i = 0; i < row * col; i++)
  {
    if (model[i] > max)
      max = model[i];

    if (model[i] < min)
      min = model[i];
  }

  return 0.01f * (max - min);
}

void directional_derivative_test(
  const float* mk,
  const float* direction,
  double chi_mk,
  double gTp,
  float alpha_0,
  SpecsContext* specs,
  data_shape_t shape
)
{
  int nx = specs->model.nx;
  int nz = specs->model.nz;

  size_t model_size = (size_t)nx * nz;

  float dt = specs->propagation.dt;

  float* m_test = malloc(model_size * sizeof(float));
  float* vp_test = malloc(model_size * sizeof(float));

  printf("\nDirectional derivative test\n");
  printf("gTp: %.15e\n", gTp);

  for (int k = 0; k < DIRECTIONAL_TEST_STEPS; ++k)
  {
    float epsilon = alpha_0 * powf(0.5f, k);

    for (size_t i = 0; i < model_size; ++i)
      m_test[i] = mk[i] + epsilon * direction[i];

    get_velocity_from_slowness(m_test, vp_test, nz, nx);

    float* dcalc_test = get_dcalc(vp_test, specs, shape);

    double chi_test = l2_norm(
      dcalc_test,
      shape.nt,
      shape.nrec,
      shape.nshot,
      dt
    );

    double directional_fd = (chi_test - chi_mk) / (double)epsilon;

    double ratio = directional_fd / gTp;

    double relative_error = fabs(directional_fd - gTp);
    relative_error /= fmax(fabs(gTp), 1e-30);

    printf(
      "epsilon: %.15e  D_fd: %.15e  ratio: %.8f  error: %.8e\n",
      (double)epsilon,
      directional_fd,
      ratio,
      relative_error
    );

    free(dcalc_test);
  }

  printf("\n");

  free(vp_test);
  free(m_test);
}

int main()
{
  PROFILE_BEGIN();

  SpecsContext* specs = Specs_Init(NULL);

  int nx = specs->model.nx;
  int nz = specs->model.nz;

  data_shape_t shape = get_data_shape(specs);

  int nt = shape.nt;
  int nrec = shape.nrec;
  int nshot = shape.nshot;

  float dt = specs->propagation.dt;
  float dh = specs->propagation.dh;

  size_t model_size = (size_t)nx * nz;
  size_t data_size = (size_t)nt * nrec * nshot;

  printf("nt: %d\n", nt);
  printf("nrec: %d\n", nrec);
  printf("nshot: %d\n", nshot);
  printf("dt: %.15e\n", (double)dt);
  printf("dh: %.15e\n", (double)dh);

  float* m_real = read2d(
    "data/FWI/marmousi_real_141x681x_dh25m.bin",
    nz,
    nx
  );
  //plot2d(m_real, nz, nx);

  float* m0 = read2d("data/FWI/m0.bin", nz, nx);
  //float* m3 = read2d("data/FWI/m_3.bin", nz, nx);
  //compare_diff(m0, m3, nz, nx, "m0", "m3");
  //plot2d(m3, nz, nx);

  float* dcalc_0 = read_any("data/FWI/dcalc_0.bin", data_size);
  //float* dcalc_0 = get_dcalc(m0, specs, shape);
  //write1d("data/FWI/dcalc_0.bin", dcalc_0, sizeof(float), data_size);

  double chi_m0 = l2_norm(dcalc_0, nt, nrec, nshot, dt);

  float* vp_current = malloc(model_size * sizeof(float));
  float* vp_k1 = malloc(model_size * sizeof(float));

  float* m_current = malloc(model_size * sizeof(float));
  float* mk1 = malloc(model_size * sizeof(float));

  float* direction = malloc(model_size * sizeof(float));

  memcpy(vp_current, m0, model_size * sizeof(float));
  get_slowness_from_velocity(vp_current, m_current, nz, nx);

  for (int it = 0; it < MAX_ITERATIONS; it++)
  {
    printf("\nIteration %d\n", it);

    // mk = m_current
    float* mk = m_current;

    // dcalc = G(m_k)
    float* dcalc_current;

    if (it == 0)
      dcalc_current = dcalc_0;
    else
      dcalc_current = get_dcalc(vp_current, specs, shape);

    // chi(m_k)
    double chi_mk = l2_norm(dcalc_current, nt, nrec, nshot, dt);

    // nabla chi(m_k)
    //float* nabla_chi;

    //if (it == 0)
    //  nabla_chi = read2d("data/FWI/nabla_chi_141x681.bin", nz, nx);
    //else
     // nabla_chi = get_nabla_gradient(vp_current, specs);

    float* nabla_chi = get_nabla_gradient(vp_current, specs);
    write2d("data/FWI/nabla_chi_141x681.bin", nabla_chi, sizeof(float), nz, nx);

    // normalized descent direction
    float grad_scale = gradient_scale(nabla_chi, model_size);

    if (grad_scale == 0.0f)
    {
      printf("Gradient is zero\n");

      free(nabla_chi);

      if (it != 0)
        free(dcalc_current);

      break;
    }

    double gTp = 0.0;

    for (size_t i = 0; i < model_size; ++i)
    {
      direction[i] = -nabla_chi[i] / grad_scale;

      gTp += (double)nabla_chi[i] * (double)direction[i];
    }

    float a_k = get_initial_alpha(mk, nz, nx);

    printf("chi_mk: %.15e\n", chi_mk);
    printf("grad_scale: %.15e\n", (double)grad_scale);
    printf("gTp: %.15e\n", gTp);
    printf("alpha_0: %.15e\n", (double)a_k);

    if (it == 0)
    {
      directional_derivative_test(
        mk,
        direction,
        chi_mk,
        gTp,
        a_k,
        specs,
        shape
      );
    }

    int accepted = 0;

    // line search
    for (int ils = 0; ils < MAX_LINE_SEARCH; ++ils)
    {
      for (size_t i = 0; i < model_size; ++i)
        mk1[i] = mk[i] + a_k * direction[i];

      get_velocity_from_slowness(mk1, vp_k1, nz, nx);
      compare_diff(m0, vp_k1, nz, nx, "m0", "vp_k1");

      float* dcalc_1 = get_dcalc(vp_k1, specs, shape);

      // chi(m_{k+1})
      double chi_mk1 = l2_norm(dcalc_1, nt, nrec, nshot, dt);

      double directional_fd = (chi_mk1 - chi_mk) / (double)a_k;
      double directional_ratio = directional_fd / gTp;

      double armijo_rhs = chi_mk + C1 * (double)a_k * gTp;

      printf("alpha: %.15e\n", (double)a_k);
      printf("chi_mk1: %.15e\n", chi_mk1);
      printf("armijo: %.15e\n", armijo_rhs);
      printf("D_fd: %.15e\n", directional_fd);
      printf("gTp: %.15e\n", gTp);
      printf("D_fd/gTp: %.8f\n", directional_ratio);

      int armijo = chi_mk1 <= armijo_rhs;

      if (armijo)
      {
        printf("ACCEPTED\n");

        compare_diff(vp_current, vp_k1, nz, nx, "vp_current", "vp_k1");

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

    if (it != 0)
      free(dcalc_current);

    if (!accepted)
    {
      printf("Line search failed at iteration %d\n", it);

      break;
    }

    save_current(vp_current, specs, it);

    if ((chi_mk / chi_m0) <= TOL)
    {
      printf("Converged at iteration %d.\n", it);

      break;
    }
  }

  free(direction);
  free(vp_k1);
  free(vp_current);
  free(mk1);
  free(m_current);
  free(dcalc_0);
  free(m0);
  free(m_real);
  free(specs);

  PROFILE_END();

  return 0;
}


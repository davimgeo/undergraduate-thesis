#include <stdlib.h>
#include <string.h>

#include <propagation.h>

#include "config/config.h"
#include "cross.objf.h"
#include "decon_objf.h"
#include "euclidian.h"
#include "fft.h"
#include "geometry.h"
#include "IO.h"
#include "plot.h"
#include "utils.h"

#define V0        2000.0f
#define REF_ALPHA 2.0f

#define SIZE      51
#define ALPHA_MIN 1.0f
#define ALPHA_MAX 3.0f

#define T0        0.5f

static float* get_model(int nz, int nx, float v0, float alpha)
{
  const size_t size = (size_t)nz * nx;

  float* model = malloc(size * sizeof(*model));
  if (model == NULL) return NULL;

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nz; ++i)
  {
    for (int j = 0; j < nx; ++j)
    {
      const size_t idx = (size_t)i * nx + j;
      model[idx] = v0 + alpha * i;
    }
  }

  return model;
}

static float* get_vel_arg(model_t* model, float dt)
{
  const size_t size = (size_t)model->nxx * model->nzz;
  const float dt2 = dt * dt;

  float* vel_arg = malloc(size * sizeof(*vel_arg));
  if (vel_arg == NULL) return NULL;

  for (size_t idx = 0; idx < size; ++idx)
  {
    const float vp = model->vp[idx];
    vel_arg[idx] = dt2 * vp * vp;
  }

  return vel_arg;
}

static void run_propagation(propagation_t* prop, seismogram_t* seis,
                            const float* vel_arg, int nt, int sidx)
{
  #pragma omp parallel
  {
    for (int t = 1; t < nt - 1; ++t)
    {
      Propagation_InjectSource(prop, sidx, t);
      Propagation_VelocityUpdate(prop, vel_arg);
      Propagation_GetSeismogram(prop, seis->seismogram, t);
    }
  }
}

static float* get_dobs(SpecsContext* specs, geometry_t* geom, wavelet_t* wave)
{
  float* base_model = get_model(
    specs->model.nz, specs->model.nx, V0, REF_ALPHA
  );

  if (base_model == NULL) return NULL;

  model_t* model = Model_Init(NULL, &specs->model);

  Model_Set(model, base_model);
  Model_Extent(model);

  seismogram_t* seis = Seismogram_Init(NULL, &specs->seismogram, geom->nrec, 0);

  propagation_t* prop = Propagation_Init(
    NULL, &specs->propagation, model, geom, wave, seis, PROPAGATION_ACOUSTIC
  );
  float* vel_arg = get_vel_arg(model, specs->propagation.dt);

  const size_t shot_size = (size_t)seis->nt * seis->nrec;
  const size_t data_size = (size_t)geom->nsrc * shot_size;

  float* dobs = malloc(data_size * sizeof(*dobs));
  if (dobs == NULL) return NULL;

  for (int shot = 0; shot < geom->nsrc; ++shot)
  {
    const int sx = geom->src.x[shot];
    const int sz = geom->src.z[shot];

    const int sidx = (sz + model->nb) * model->nxx + (sx + model->nb);

    //Propagation_ResetFields(prop);

    run_propagation(prop, seis, vel_arg, seis->nt, sidx);

    memcpy(
      dobs + (size_t)shot * shot_size,
      seis->seismogram,
      shot_size * sizeof(*dobs)
    );
  }

  free(vel_arg);

  return dobs;
}

int main(void)
{
  PROFILE_BEGIN();

  SpecsContext* specs = Specs_Init(NULL);

  geometry_t* geom = Geometry_InitCreate(NULL, &specs->geometry);

  Geometry_Create(geom, GEOMETRY_ONLY_RECEIVERS);
  Geometry_SetSource(geom, 101, 20);
  Geometry_SetSource(geom, 151, 20);
  Geometry_SetSource(geom, 51, 20);

  wavelet_t* wave = Wavelet_Init(NULL, &specs->wavelet);
  Wavelet_Create(wave);

  float* dobs = get_dobs(specs, geom, wave);
  if (dobs == NULL) return -1;

  float* alphas = malloc(SIZE * sizeof(float));
  float* l2 = malloc(SIZE * sizeof(float));
  float* l1 = malloc(SIZE * sizeof(float));
  float* cross = malloc(SIZE * sizeof(float));
  float* decon = malloc(SIZE * sizeof(float));

  if (alphas == NULL || l2 == NULL || l1 == NULL || cross == NULL ||
      decon == NULL)
  {
    free(alphas);
    free(l2);
    free(l1);
    free(cross);
    free(decon);
    free(dobs);

    return -1;
  }

  const float da = (ALPHA_MAX - ALPHA_MIN) / (float)(SIZE - 1);

  for (int i = 0; i < SIZE; ++i)
  {
    alphas[i] = ALPHA_MIN + i * da;

    float* grad_model = get_model(
      specs->model.nz, specs->model.nx, V0, alphas[i]
    );

    if (grad_model == NULL) return -1;

    model_t* grad_model_obj = Model_Init(NULL, &specs->model);

    Model_Set(grad_model_obj, grad_model);
    Model_Extent(grad_model_obj);

    float* vel_arg = get_vel_arg(grad_model_obj, specs->propagation.dt);
    if (vel_arg == NULL) return -1;

    seismogram_t* seis_grad = Seismogram_Init(
      NULL, &specs->seismogram, geom->nrec, 0);

    propagation_t* prop_grad = Propagation_Init(
      NULL, &specs->propagation, grad_model_obj, geom, wave, seis_grad,
      PROPAGATION_ACOUSTIC
    );

    const int nt = seis_grad->nt;
    const int nrec = seis_grad->nrec;
    const size_t shot_size = (size_t)nt * nrec;

    l2[i] = 0.0f;
    l1[i] = 0.0f;
    cross[i] = 0.0f;
    decon[i] = 0.0f;

    for (int shot = 0; shot < geom->nsrc; ++shot)
    {
      const int sx = geom->src.x[shot];
      const int sz = geom->src.z[shot];

      const int sidx =
        (sz + grad_model_obj->nb) * grad_model_obj->nxx +
        (sx + grad_model_obj->nb);

      //Propagation_ResetFields(prop_grad);

      run_propagation(prop_grad, seis_grad, vel_arg, nt, sidx);

      float* dcalc = seis_grad->seismogram;
      float* dobs_shot = dobs + (size_t)shot * shot_size;

      l2[i] += l2_norm_2d(dcalc, dobs_shot, nt, nrec);
      l1[i] += l1_norm_2d(dcalc, dobs_shot, nt, nrec);

      cross[i] += get_cross_2d(
        dcalc, dobs_shot, seis_grad->dt, nt, nrec, T0
      );

      decon[i] += get_decon_2d(
        dcalc, dobs_shot, seis_grad->dt, nt, nrec, T0
      );
    }

    if (i == 1) plot_model_geometry(grad_model_obj, 10, geom);

    if (i == SIZE - 1) plot_model_geometry(grad_model_obj, 10, geom);

    printf(
      "Alpha: %g | L2: %g | L1: %g | Cross: %g | Decon: %g\n",
      alphas[i], l2[i], l1[i], cross[i], decon[i]
    );

    progress_bar(i, SIZE);

    free(vel_arg);
  }

  normalize(l2, SIZE);
  normalize(l1, SIZE);
  normalize(cross, SIZE);
  normalize(decon, SIZE);

  PROFILE_END();

  write1d("data/l2_51.bin", l2, sizeof(float), SIZE);
  write1d("data/l1_51.bin", l1, sizeof(float), SIZE);
  write1d("data/cross_51.bin", cross, sizeof(float), SIZE);
  write1d("data/decon_51.bin", decon, sizeof(float), SIZE);
  write1d("data/alphas_51.bin", alphas, sizeof(float), SIZE);

  plot1d_xy(alphas, l2, SIZE);
  plot1d_xy(alphas, l1, SIZE);
  plot1d_xy(alphas, cross, SIZE);
  plot1d_xy(alphas, decon, SIZE);

  free(alphas);
  free(l2);
  free(l1);
  free(cross);
  free(decon);
  free(dobs);

  return 0;
}

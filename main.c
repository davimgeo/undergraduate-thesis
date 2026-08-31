#include <stdlib.h>
#include <propagation.h>
#include "cross.objf.h"
#include "decon_objf.h"
#include "euclidian.h"
#include "fft.h"
#include "geometry.h"
#include "utils.h"
#include "config/config.h"
#include "plot.h"
#include "IO.h"

#define DH 10.0f

#define REF_V0      2000.0f
#define REF_ALPHA   0.7f

#define SIZE        11
#define ALPHA_MIN   0.45f
#define ALPHA_MAX   0.8f
#define V0_MIN      1500.0f
#define V0_MAX      2500.0f

#define T0          0.5f

float* get_model(int nz, int nx, int v0, float alpha)
{
  float* model = (float*)malloc(nz * nx * sizeof(float));
  if (model == NULL) return NULL;

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nz; i++)
  {
    for (int j = 0; j < nx; j++)
    {
      model[i * nx + j] = v0 + alpha * i * DH;
    }
  }

  return model;
}

float* get_dobs(
  SpecsContext* specs,
  geometry_t* geom,
  wavelet_t* wave
)
{
  float* base_model = get_model(
    specs->model.nz,
    specs->model.nx,
    REF_V0,
    REF_ALPHA
  );

  model_t* model = Model_Init(NULL, &specs->model);
  Model_Set(model, base_model);
  Model_Extent(model);

  seismogram_t* seis = Seismogram_Init(
    NULL,
    &specs->seismogram,
    geom->nrec, 0
  );

  propagation_t* prop = Propagation_Init(
    NULL,
    &specs->propagation,
    model,
    geom,
    wave,
    seis,
    PROPAGATION_ACOUSTIC
  );

  Propagation_Run(prop, 0);

  float* dobs = seis->seismogram;
  plot_model_geometry(model, DH, geom);
  return dobs;
}

int main()
{
  PROFILE_BEGIN();

  SpecsContext* specs = Specs_Init(NULL);

  geometry_t* geom = Geometry_InitCreate(NULL, &specs->geometry);
  Geometry_Create(geom, GEOMETRY_ONLY_RECEIVERS);
  Geometry_SetSource(geom, 850, 50);

  wavelet_t* wave = Wavelet_Init(NULL, &specs->wavelet);
  Wavelet_Create(wave);

  float* dobs = get_dobs(specs, geom, wave);

  float* alphas = malloc(SIZE * sizeof(float));
  float* v0     = malloc(SIZE * sizeof(float));

  float* l2    = malloc(SIZE * SIZE * sizeof(float));
  float* l1    = malloc(SIZE * SIZE * sizeof(float));
  float* cross = malloc(SIZE * SIZE * sizeof(float));
  float* decon = malloc(SIZE * SIZE * sizeof(float));

  if (alphas == NULL || v0 == NULL || l2 == NULL || l1 == NULL ||
      cross == NULL || decon == NULL)
  {
    free(alphas);
    free(v0);
    free(l2);
    free(l1);
    free(cross);
    free(decon);

    return -1;
  }

  float da = (ALPHA_MAX - ALPHA_MIN) / (float)(SIZE - 1);
  float dv = (V0_MAX - V0_MIN) / (float)(SIZE - 1);

  for (int i = 0; i < SIZE; i++)
  {
    v0[i] = V0_MIN + i * dv;
    alphas[i] = ALPHA_MIN + i * da;
  }

  for (int i = 0; i < SIZE; i++)
  {
    for (int j = 0; j < SIZE; j++)
    {
      int idx = i * SIZE + j;

      float* grad_model = get_model(
        specs->model.nz,
        specs->model.nx,
        v0[i],
        alphas[j]
      );

      model_t* grad_model_obj = Model_Init(NULL, &specs->model);
      Model_Set(grad_model_obj, grad_model);
      Model_Extent(grad_model_obj);

      seismogram_t* seis_grad = Seismogram_Init(
        NULL,
        &specs->seismogram,
        geom->nrec,
        0
      );

      propagation_t* prop_grad = Propagation_Init(
        NULL,
        &specs->propagation,
        grad_model_obj,
        geom,
        wave,
        seis_grad,
        PROPAGATION_ACOUSTIC
      );

      Propagation_Run(prop_grad, 0);

      float* dcalc = seis_grad->seismogram;
      int nt = seis_grad->nt;
      int nrec = seis_grad->nrec;
      //plot_seismogram(seis_grad, geom->offset_rec);
      //plot2d(dobs, seis_grad->nt, seis_grad->nrec);

      l2[idx] = l2_squared_norm_2d(
        dcalc,
        dobs,
        nt,
        nrec
      );

      l1[idx] = l1_norm_2d(
        dcalc,
        dobs,
        nt,
        nrec
      );

      cross[idx] = get_cross_2d(
        dcalc,
        dobs,
        seis_grad->dt,
        nt,
        geom->nrec,
        T0
      );

      decon[idx] = get_decon_2d(
        dcalc,
        dobs,
        seis_grad->dt,
        nt,
        geom->nrec,
        T0
      );

      printf(
        "V0: %g | Alpha: %g | L2: %g | L1: %g | Cross: %g | Decon: %g\n",
        v0[i],
        alphas[j],
        l2[idx],
        l1[idx],
        cross[idx],
        decon[idx]
      );

      //Propagation_Destroy(prop_grad);
      //Seismogram_Destroy(seis_grad);
      //Model_Destroy(grad_model_obj);
      //free(grad_model);

      progress_bar(idx, SIZE * SIZE);
    }
  }

  normalize(l2, SIZE * SIZE);
  normalize(l1, SIZE * SIZE);
  normalize(cross, SIZE * SIZE);
  normalize(decon, SIZE * SIZE);

  PROFILE_END();

  write1d("data/l2.bin", l2, sizeof(float), SIZE * SIZE);
  write1d("data/l1.bin", l1, sizeof(float), SIZE * SIZE);
  write1d("data/cross.bin", cross, sizeof(float), SIZE * SIZE);
  write1d("data/decon.bin", decon, sizeof(float), SIZE * SIZE);

  write1d("data/alphas.bin", alphas, sizeof(float), SIZE);
  write1d("data/v0.bin", v0, sizeof(float), SIZE);

  plot3d(decon, alphas, v0, SIZE, SIZE);
  plot3d(cross, alphas, v0, SIZE, SIZE);
  plot3d(l2, alphas, v0, SIZE, SIZE);
  plot3d(l1, alphas, v0, SIZE, SIZE);

  free(alphas);
  free(v0);
  free(l2);
  free(l1);
  free(cross);
  free(decon);

  return 0;
}


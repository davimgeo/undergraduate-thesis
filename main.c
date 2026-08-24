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

#define V0          2000.0f
#define REF_ALPHA   2.0f

#define SIZE        51
#define ALPHA_MIN   1.0f
#define ALPHA_MAX   3.0f

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
      model[i * nx + j] = v0 + alpha * i;
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
    V0,
    REF_ALPHA
  );

  model_t* model = Model_Init(NULL, &specs->model);
  Model_Set(model, base_model);
  Model_Extent(model);

  seismogram_t* seis = Seismogram_Init(
    NULL,
    &specs->seismogram,
    geom->nrec
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
  Propagation_GetDamp(prop);
  Propagation_Run(prop, 0);

  float* dobs = seis->seismogram;
  return dobs;
}

int main()
{
  PROFILE_BEGIN();

  SpecsContext* specs = Specs_Init(NULL);

  geometry_t* geom = Geometry_InitCreate(NULL, &specs->geometry);
  Geometry_Create(geom, GEOMETRY_ONLYRECEIVERS);
  //Geometry_SetReceiver(geom, 101, 0);
  Geometry_SetSource(geom, 101, 50);

  wavelet_t* wave = Wavelet_Init(NULL, &specs->wavelet);
  Wavelet_Create(wave);

  float* dobs = get_dobs(specs, geom, wave);

  float* alphas = malloc(SIZE * sizeof(float));
  float* l2 = malloc(SIZE * sizeof(float));
  float* l1 = malloc(SIZE * sizeof(float));
  float* cross = malloc(SIZE * sizeof(float));
  float* decon = malloc(SIZE * sizeof(float));

  if (alphas == NULL || l2 == NULL || l1 == NULL ||
      cross == NULL || decon == NULL)
  {
    free(alphas);
    free(l2);
    free(l1);
    free(cross);
    free(decon);
    return -1;
  }

  for (int i = 0; i < SIZE; i++)
  {
    float da = (ALPHA_MAX - ALPHA_MIN) / (float)(SIZE - 1);
    alphas[i] = ALPHA_MIN + i * da;

    float* grad_model = get_model(
      specs->model.nz,
      specs->model.nx,
      V0,
      alphas[i]
    );

    model_t* grad_model_obj = Model_Init(NULL, &specs->model);
    Model_Set(grad_model_obj, grad_model);
    Model_Extent(grad_model_obj);

    seismogram_t* seis_grad = Seismogram_Init(
      NULL, &specs->seismogram, geom->nrec
    );

    propagation_t* prop_grad = Propagation_Init(
      NULL, &specs->propagation, grad_model_obj, geom,
      wave, seis_grad, PROPAGATION_ACOUSTIC
    );

    Propagation_GetDamp(prop_grad);
    Propagation_Run(prop_grad, 0);

    if(i == 1) plot_model_geometry(grad_model_obj, 10, geom);
    if(i == SIZE - 1) plot_model_geometry(grad_model_obj, 10, geom);

    float* dcalc = seis_grad->seismogram;
    int nt = seis_grad->nt;
    int nrec = seis_grad->nrec;

    l2[i] = l2_norm_2d(dcalc, dobs, nt, nrec);
    l1[i] = l1_norm_2d(dcalc, dobs, nt, nrec);

    cross[i] = get_cross_2d(
      dcalc, dobs, seis_grad->dt, nt, geom->nrec, T0
    );

    decon[i] = get_decon_2d(
      dcalc, dobs, seis_grad->dt, nt, geom->nrec, T0
    );

    printf("Alpha: %g | L2: %g | L1: %g | Cross: %g | Decon: %g\n",
      alphas[i], l2[i], l1[i], cross[i], decon[i]);

    progress_bar(i, SIZE);
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

  return 0;
}

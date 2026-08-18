#include <fftw3.h>
#include <stdlib.h>
#include <propagation.h>
#include "cross.objf.h"
#include "fft.h"
#include "decon_objf.h"
#include "euclidian.h"
#include "utils.h"
#include "config/config.h"
#include "plot.h"
#include "IO.h"

#define INTERFACE   250
#define LAYER_VALUE 2500.0f
#define V0          2000.0f
#define REF_ALPHA   0.5

#define SIZE        51
#define ALPHA_MIN   0
#define ALPHA_MAX   1.0f

#define T0          50

float* get_model_layer(
  int nz,
  int nx,
  int v0,
  float alpha,
  int interface,
  float layer_value
)
{
  float* model = (float*)malloc(nz * nx * sizeof(float));
  if (model == NULL) return NULL;

  for (int i = 0; i < nz; i++)
  {
    float velocity;

    if (i <= interface)
    {
      //float z = (float)i / (float)(interface - 1);

      //velocity = v0 + alpha * (layer_value - v0) * z;
      velocity = v0;
    }
    else
    {
      velocity = layer_value;
    }

    for (int j = 0; j < nx; j++)
        model[i * nx + j] = velocity;
  }

  return model;
}

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
  Geometry_SetSource(geom, 101, 50);

  wavelet_t* wave = Wavelet_Init(NULL, &specs->wavelet);
  Wavelet_Create(wave);

  float* dobs = get_dobs(specs, geom, wave);

  float* alphas = (float*)malloc(sizeof(float) * SIZE);
  if (alphas == NULL) return -1;
  float* cross = (float*)malloc(SIZE * sizeof(float));
  if (cross == NULL) return -1;

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

    if(i == 1) plot_model_geometry(grad_model_obj, 10, geom);
    if(i == SIZE - 1) plot_model_geometry(grad_model_obj, 10, geom);

    seismogram_t* seis_grad = Seismogram_Init(NULL, &specs->seismogram, geom->nrec);

    propagation_t* prop_grad = Propagation_Init(
      NULL,
      &specs->propagation,
      grad_model_obj,
      geom,
      wave,
      seis_grad,
      PROPAGATION_ACOUSTIC
    );
    Propagation_GetDamp(prop_grad);
    Propagation_Run(prop_grad, 0);

    float* dcalc = seis_grad->seismogram;

    cross[i] = get_cross_2d(
       dcalc, dobs, seis_grad->dt,
        seis_grad->nt, seis_grad->nrec, T0
    );

    printf("Objective Function: %g, Alpha: %g\n", cross[i], alphas[i]);

    progress_bar(i, SIZE);
  }

  normalize(cross, SIZE);

  PROFILE_END();

  plot1d(cross, SIZE);

  write1d("data/cross_31.bin", cross, sizeof(float), SIZE);
  write1d("data/alphas_31.bin", alphas, sizeof(float), SIZE);

  return 0;
}

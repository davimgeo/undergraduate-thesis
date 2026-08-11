#include <stdlib.h>
#include <propagation.h>

#include "1D/cross.objf.h"
#include "1D/euclidian.h"
#include "utils.h"
#include "config/config.h"
#include "plot.h"
#include "IO.h"

#define NZ          201
#define NX          201
#define V0          2500
#define REF_ALPHA   1.2f

#define SIZE        51
#define ALPHA_MIN   0.6f
#define ALPHA_MAX   5.0f

float* get_model(int nz, int nx, int v0, float alpha)
{
  float* model = (float*)malloc(nz * nx * sizeof(float));
  if (model == NULL) return NULL;

  for (int i = 0; i < nz; i++)
    for (int j = 0; j < nx; j++)
      model[i * nx + j] = v0 + i / alpha;

  return model;
}

float* get_dobs(
  SpecsContext* specs,
  geometry_t* geom,
  wavelet_t* wave
)
{
  float* base_model = get_model(
    NZ,
    NX,
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

  geometry_t* geom = Geometry_InitCreate(
    NULL,
    &specs->geometry
  );
  Geometry_Create(geom, GEOMETRY_ONLYRECEIVERS);
  Geometry_SetSource(geom, 108, 30);

  wavelet_t* wave = Wavelet_Init(
    NULL,
    &specs->wavelet
  );
  Wavelet_Create(wave);

  float* dobs = get_dobs(specs, geom, wave);

  float* result = (float*)malloc(SIZE * sizeof(float));
  if (result == NULL) return -1;

  for (int i = 0; i < SIZE; i++)
  {
    float da = (ALPHA_MAX - ALPHA_MIN) / SIZE;
    float alpha = ALPHA_MIN + i * da;

    float* grad_model = get_model(
      NZ,
      NX,
      V0,
      alpha
    );

    model_t* grad_model_obj = Model_Init(
      NULL,
      &specs->model
    );

    Model_Set(grad_model_obj, grad_model);
    Model_Extent(grad_model_obj);

    seismogram_t* seis_grad = Seismogram_Init(
      NULL,
      &specs->seismogram,
      geom->nrec
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
    Propagation_GetDamp(prop_grad);
    Propagation_Run(prop_grad, 0);

    float* dcalc = seis_grad->seismogram;

    // not working, forgot I only did for 1d case
    result[i] = get_cross_result(
        dobs, dcalc, seis_grad->dt, seis_grad->nt, 30
    );
  }

  PROFILE_END();

  plot1d(result, SIZE);

  write1d("data/l1_norm_51.bin", result, sizeof(float), SIZE);

  return 0;
}

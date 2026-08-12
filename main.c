#include <stdlib.h>
#include <propagation.h>

#include "cross.objf.h"
#include "decon_objf.h"
#include "euclidian.h"
#include "utils.h"
#include "config/config.h"
#include "plot2.h"
#include "plot.h"
#include "IO.h"

#define NZ          1001
#define NX          501
#define V0          2000
#define REF_ALPHA   0.3

#define SIZE        41
#define ALPHA_MIN   -1.0f
#define ALPHA_MAX   1.0f

#define T0          50

float* get_model(int nz, int nx, int v0, float alpha)
{
  float* model = (float*)malloc(nz * nx * sizeof(float));
  if (model == NULL) return NULL;

  for (int i = 0; i < nz; i++)
  {
    for (int j = 0; j < nx; j++)
    {
      float dz = (nz - (float)i) / nz;
      model[i * nx + j] = v0 + alpha * i * dz;
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

  geometry_t* geom = Geometry_InitCreate(NULL, &specs->geometry);
  Geometry_Create(geom, GEOMETRY_ONLYRECEIVERS);
  Geometry_SetSource(geom, 501, 30);

  wavelet_t* wave = Wavelet_Init(NULL, &specs->wavelet);
  Wavelet_Create(wave);

  float* dobs = get_dobs(specs, geom, wave);

  float* alphas = (float*)malloc(sizeof(float) * SIZE);
  float* result = (float*)malloc(SIZE * sizeof(float));
  if (result == NULL) return -1;

  for (int i = 0; i < SIZE; i++)
  {
    float da = (ALPHA_MAX - ALPHA_MIN) / SIZE;
    alphas[i] = ALPHA_MIN + i * da;

    float* grad_model = get_model(NZ, NX, V0, alphas[i]);

    model_t* grad_model_obj = Model_Init(NULL, &specs->model);

    Model_Set(grad_model_obj, grad_model);
    Model_Extent(grad_model_obj);

    if(i == 1) 
    {
      model_t* m = grad_model_obj;
      plot_model_geometry(
        m->vp,
        m->nx,
        m->nz,
        m->nb,
        specs->propagation.dh, 
        geom->rec.x,
        geom->rec.z,
        geom->nrec,
        geom->src.x,
        geom->src.z,
        geom->nsrc
      );
    }

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

    result[i] = get_cross_2d(
        dcalc, dobs, seis_grad->dt, 
        seis_grad->nt, seis_grad->nrec, T0
    );

    //result[i] = get_decon_2d(
    //    dcalc, dobs, seis_grad->dt,
    //    seis_grad->nt, seis_grad->nrec, T0
    //);

    //result[i] = l2_norm_2d(dcalc, dobs, seis_grad->nt, seis_grad->nrec);

    printf("Objective Function: %g, Alpha: %g\n", result[i], alphas[i]);

    progress_bar(i, SIZE);
  }

  normalize(result, SIZE);

  PROFILE_END();

  plot1d(result, SIZE);

  //write1d("data/decon_51.bin", result, sizeof(float), SIZE);
  //write1d("data/alphas.bin", alphas, sizeof(float), SIZE);

  return 0;
}

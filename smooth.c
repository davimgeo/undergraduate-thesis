#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "config/config.h"
#include "propagation.h"
#include "plot.h"
#include "fft.h"

#define PI 3.1415f

float* extent_model(float* model, int nx, int nz, int nb)
{
  int nxx = nx + 2*nb;
  int nzz = nz + 2*nb;

  float *model_ext = calloc(sizeof(float), nxx * nzz);

  /* copy original arr into ext */
  for (int j = 0; j < nx; j++) 
  {
    for (int i = 0; i < nz; i++) 
    {
      model_ext[(i + nb) * nxx + (j + nb)]  = model[i * nx + j];
    }
  }

  /* pad bottom and upper*/
  for (int j = nb; j < nx+nb; j++) 
  {
    for (int i = 0; i < nb; i++) 
    {
      // bottom
      model_ext[i * nxx + j] = model_ext[nb * nxx + j];

      // up
      model_ext[(nz + nb + i) * nxx + j] = model_ext[(nz + nb - 1) * nxx + j];
    }
  }

  /* pad left and right respectively */
  for (int i = 0; i < nzz; i++) 
  {
    for (int j = 0; j < nb; j++) 
    {
      // counld vectorize because of strided loop
      model_ext[i * nxx + j]  = model_ext[i * nxx + nb];

      model_ext[i * nxx + (nx + nb + j)] = model_ext[i * nxx + (nx + nb - 1)];
    }
  }

  /* swap pointers to new arr */
  return model_ext;
}

float* gaussian_filter_2d(
  float* model,
  float dz, float dx,
  float w1, float w2,
  float alpha,
  int row, int col
)
{
  int nb = 30;

  int row_ext = row + 2*nb;
  int col_ext = col + 2*nb;

  float* model_ext = extent_model(model, col, row, nb);

  float* k1 = calloc(row_ext, sizeof(float));
  float* k2 = calloc(col_ext, sizeof(float));

  for (int i = 1; i <= row_ext / 2; i++)
  {
    k1[i] = ((float)i / (row_ext - 1)) / dz;
    k1[row_ext - i] = -k1[i];
  }

  for (int j = 1; j <= col_ext / 2; j++)
  {
    k2[j] = ((float)j / (col_ext - 1)) / dx;
    k2[col_ext - j] = -k2[j];
  }

  _Complex float* C_model = get_fft_2d(
    model_ext,
    row_ext,
    col_ext
  );

  for (int i = 0; i < row_ext; i++)
  {
    for (int j = 0; j < col_ext; j++)
    {
      float filter =
        expf(-alpha * k1[i] * k1[i] / (w1 * w1)) *
        expf(-alpha * k2[j] * k2[j] / (w2 * w2));

      C_model[i * col_ext + j] *= filter;
    }
  }

  float* result_ext = get_ifft_2d(C_model, row_ext, col_ext);

  float* result = malloc(row * col * sizeof(float));

  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      result[i * col + j] =
        result_ext[(i + nb) * col_ext + (j + nb)];
    }
  }

  free(k1);
  free(k2);
  free(model_ext);
  free(C_model);
  free(result_ext);

  return result;
}

float* smooth(
  const float* model,
  int kernel_size,
  float sigma,
  int row,
  int col
)
{
  int ks = kernel_size;
  int half = ks / 2;

  float* gaussian_kernel = malloc(ks * ks * sizeof(float));

  float kernel_sum = 0.0f;

  for (int k = 0; k < ks; k++)
  {
    for (int l = 0; l < ks; l++)
    {
      float x = k - half;
      float y = l - half;

      float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));

      gaussian_kernel[k * ks + l] = value;
      kernel_sum += value;
    }
  }

  for (int k = 0; k < ks; k++)
  {
    for (int l = 0; l < ks; l++)
    {
      gaussian_kernel[k * ks + l] /= kernel_sum;
    }
  }

  float* result = malloc(row * col * sizeof(float));

  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      float temp = 0.0f;
      float weight_sum = 0.0f;

      for (int k = -half; k <= half; k++)
      {
        for (int l = -half; l <= half; l++)
        {
          int ii = i - k;
          int jj = j - l;

          if (ii < 0 || ii >= row || jj < 0 || jj >= col) continue;

          float weight = gaussian_kernel[(k + half) * ks + (l + half)];

          temp += weight * model[ii * col + jj];
          weight_sum += weight;
        }
      }

      result[i * col + j] = temp / weight_sum;
    }
  }

  free(gaussian_kernel);

  return result;
}

float* running_mean(
  const float* model,
  int kernel_size,
  int row,
  int col
)
{
  int ks = kernel_size;
  int half = ks / 2;

  float* result = malloc(row * col * sizeof(float));

  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      float temp = 0.0f;
      int count = 0;

      for (int k = -half; k <= half; k++)
      {
        for (int l = -half; l <= half; l++)
        {
          int ii = i - k;
          int jj = j - l;

          if (ii < 0 || ii >= row || jj < 0 || jj >= col) continue;

          temp += model[ii * col + jj];
          count++;
        }
      }

      result[i * col + j] = temp / (float)count;
    }
  }

  return result;
}

int main()
{
  int kernel_size = 70;
  
  int dh = 25;

  SpecsContext* specs = Specs_Init(NULL);

  model_t* model = Model_Init(NULL, &specs->model);
  Model_Load(model, "data/vp_351x1701_10m.bin", 1701, 351);

  //float* vp_smooth = running_mean(model->vp, kernel_size, nz, nx);
  float* vp_smooth = gaussian_filter_2d(model->vp, dh, dh, 2e-3f, 3e-4f, 3.0f, model->nz, model->nx);

  plot2d(vp_smooth, model->nz, model->nx);

  return 0;
}

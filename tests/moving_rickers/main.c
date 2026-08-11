#include <stdlib.h>
#include <string.h>

#include "IO.h"

#include "1D/ricker.h"
#include "1D/moving_rickers.h"

#include "plot.h"

int main()
{
    int nt = 1001;
    int result_size = 200;

    float fmax = 30.0f;
    float dt = 1e-3f;

    float t0 = 200;

    float* cross_all = (float*)malloc(result_size * result_size * sizeof(float));
    float* l1_all    = (float*)malloc(result_size * result_size * sizeof(float));
    float* l2_all    = (float*)malloc(result_size * result_size * sizeof(float));
    float* decon_all = (float*)malloc(result_size * result_size * sizeof(float));

    float mid = (nt * dt) / 2;

    for(int i = 0; i < result_size; i++)
    {
      float lag1 = 0.6f - i * ((0.6f - mid) / result_size);
      float lag2 = 0.05f + i * ((mid - 0.05f) / result_size);

      float* ricker1 = get_ricker(nt, fmax, dt, lag1);
      float* ricker2 = get_ricker(nt, fmax, dt, lag2);

      float* ricker = (float*)malloc(nt * sizeof(float));
      for(int t = 0; t < nt; t++) ricker[t] = ricker1[t] + ricker2[t];

      float* cross = moving_rickers_cross(ricker, nt, result_size, fmax, dt, t0);
      float* l1    = moving_rickers_l1(ricker, nt, result_size);
      float* l2    = moving_rickers_l2(ricker, nt, result_size);
      float* decon = moving_rickers_decon(ricker, nt, result_size, fmax, dt, t0);

      memcpy(cross_all + i * result_size, cross, result_size * sizeof(float));
      memcpy(l1_all    + i * result_size, l1,    result_size * sizeof(float));
      memcpy(l2_all    + i * result_size, l2,    result_size * sizeof(float));
      memcpy(decon_all + i * result_size, decon, result_size * sizeof(float));

      free(ricker1);
      free(ricker2);
      free(ricker);
      free(cross);
      free(l1);
      free(l2);
      free(decon);

      printf("Progress: %.1f%%\n", 100.0f * (float)(i + 1) / result_size);
      printf("\e[1;1H\e[2J");
    }

    write1d("data/temp/cross.bin", cross_all, sizeof(float), result_size * result_size);
    write1d("data/temp/l1.bin",    l1_all,    sizeof(float), result_size * result_size);
    write1d("data/temp/l2.bin",    l2_all,    sizeof(float), result_size * result_size);
    write1d("data/temp/decon.bin", decon_all, sizeof(float), result_size * result_size);

    free(cross_all);
    free(l1_all);
    free(l2_all);
    free(decon_all);

    return 0;
}

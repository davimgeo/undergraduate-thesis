#include <stdlib.h>

#include "cross.objf.h"
#include "decon_objf.h"

#include "euclidian.h"
#include "plot.h"
#include "IO.h"
#include "utils.h"

#define NT 2001
#define DT 1e-3f
#define T0 0.1f

int main()
{
  float* dobs = read1d("data/dobs.bin", NT);

  float* dcalc1 = read1d("data/dcalc_2.5f.bin", NT);
  float* dcalc2 = read1d("data/dcalc_3.0f.bin", NT);
  float* dcalc3 = read1d("data/dcalc_3.5f.bin", NT);
  float* dcalc4 = read1d("data/dcalc_4.0f.bin", NT);
  float* dcalc5 = read1d("data/dcalc_4.5f.bin", NT);

  float* dcalc[5] = {dcalc1, dcalc2, dcalc3, dcalc4,dcalc5};

  float cross[5];
  float l2[5];

  for (int i = 0; i < 5; ++i)
  {
    cross[i] = get_cross_1d(dcalc[i], dobs, DT, NT, T0);
    l2[i] = l2_norm_1d(dobs, dcalc[i], NT);
  }

  normalize(cross, 5);
  normalize(l2, 5);

  plot1d_compare(cross, l2, 5);

  free(dobs);

  free(dcalc1);
  free(dcalc2);
  free(dcalc3);
  free(dcalc4);
  free(dcalc5);

  return 0;
}

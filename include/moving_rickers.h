#pragma once

float* moving_rickers_l1(
  float* wavelet,
  int nt,
  int result_size
);

float* moving_rickers_l2(
float* wavelet,
int nt,
int result_size
);

float* moving_rickers_cross(
  float* u_o,
  int nt,
  int result_size,
  float fmax,
  float dt,
  float t0
);

float* moving_rickers_decon(
  float* u_o,
  int nt,
  int result_size,
  float fmax,
  float dt,
  float t0
);

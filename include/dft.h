#pragma once

#include <complex.h>

float complex* computeDFT(int N, float* arr, float dt);

float* computeIFFT(int N, float complex* X);

float* dftshift(float* x, int N);

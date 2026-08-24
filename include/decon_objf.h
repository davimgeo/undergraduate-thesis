#pragma once

float get_decon_1d(float *dcalc, float *dobs, float dt, int nt, float t0);
float* get_d_1d(float* dcalc, float* dobs, float dt, int nt);
float get_decon_2d(float *dcalc, float *dobs, float dt, int nt, int nrec, float t0);

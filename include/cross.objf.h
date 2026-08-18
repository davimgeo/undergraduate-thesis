#pragma once

float get_cross_1d(float *u_s, float *u_o, float dt, int nt, float t0);

float* get_c_2d(float* u_s, float* u_o, float dt, int nt, int nrec);
float get_cross_2d(float *u_s, float *u_o, float dt, int nt, int nrec, float t0);

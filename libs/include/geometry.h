#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <stdio.h>

#define GEOMETRY_ONLYRECEIVERS (1U << 0)

typedef struct
{
  float* x;
  float* z;
} receivers_t;

typedef struct
{
  float* x;
  float* z;
} sources_t;

typedef struct
{
  int line_length;
  int src_depth;
  int rec_depth;
  int offset_rec;
  int offset_src;
} geometry_specs_t;

typedef struct geometry_t
{
  receivers_t rec;
  sources_t src;

  int line_length;
  int src_depth;
  int rec_depth;
  int offset_rec;
  int offset_src;

  int dh;

  int nrec, nsrc;
} geometry_t;

geometry_t* Geometry_InitLoad(geometry_t* g);

geometry_t* Geometry_InitCreate(geometry_t* g, geometry_specs_t* specs);

void Geometry_Load(
  geometry_t* geom, 
  const char* REC_PATH, 
  const char* SRC_PATH
);

void Geometry_SetSource(geometry_t* geom, int sx, int sz);
void Geometry_SetReceiver(geometry_t* geom, int rx, int rz);
void Geometry_Create(geometry_t *geom, unsigned flags);

#endif /* end of include guard: GEOMETRY_H */


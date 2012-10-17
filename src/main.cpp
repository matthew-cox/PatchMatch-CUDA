// Copyright 2009 Matthew Cox and Earl Kinney
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include "def.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "bmploader.h"
#include "nn.h"
#include "nn_gpu.hpp"

/* main for reconstruction code */
int main ( int argc, char **argv ) {
  if ( argc < 2 ) {

    printf( "Usage: reconstruct input1.bmp input2.bmp\n" );
    exit(1);
  }

  if ( NNF_DEBUG ){ printf( "A mapent is %i bytes\n", (int)sizeof( mapent ) ); }
  // seed random
  srand( time( NULL ) );

  /******************************************************************************/
  // load input BMP file(s)

  int fWidth, fHeight, sWidth, sHeight;
  uchar4 *h_fst, *h_snd; 

  LoadBMPFile( &h_fst, &fWidth, &fHeight, argv[1] );

  LoadBMPFile( &h_snd, &sWidth, &sHeight, argv[2] );

  float totalTime = 0;
  float maxErr = 0;
  unsigned int timer = 0;
  printf( "CPU Timer starts...\n" );
  cutilCheckError( cutCreateTimer( &timer ) );
  cutilCheckError( cutResetTimer( timer ) );
  cutilCheckError( cutStartTimer( timer ) );


  /******************************************************************************/
  int fPixels = fWidth * fHeight;
  //int sPixels = sWidth * sHeight;

  // randomally allocate initial pixel map - size of original image
  // FIXME: what to do for resizing?
  mapent *h_map;
  h_map = (mapent*) malloc( fPixels * sizeof( mapent ) );

  nn_random_map( h_fst, fWidth, fHeight, h_snd, sWidth, sHeight, h_map );

  // visualize the random map
  if ( NNF_VIZ ) {
    //cutilCheckError( cutStopTimer( timer ) );
    nn_viz_map( fWidth, fHeight, h_snd, sWidth, sHeight, h_map, "cpu-iter-0.bmp" );
    //cutilCheckError( cutStartTimer( timer ) );
  }

  /******************************************************************************/
  // CPU iteration loop

  for ( int i = 1; i <= ITERATIONS; i++ ) {

    if ( NNF_DEBUG ){ printf( "Starting iteration %i...\n", i ); }

    mapent *newMap;
    newMap = (mapent*) malloc( fPixels * sizeof( mapent ) );

    // do the search
    nn_search( h_fst, fWidth, fHeight, h_snd, sWidth, sHeight, h_map, newMap );

    // store the results
    free( h_map );
    h_map = newMap;

    // viz the iterations?
    if ( NNF_VIZ ) {
      char fname[64];
      sprintf( fname, "cpu-iter-%i.bmp", i );

      if ( NNF_DEBUG ){ printf( "Saving iteration %i output...\n", i ); }
      nn_viz_map( fWidth, fHeight, h_snd, sWidth, sHeight, h_map, fname );
    }
  }

  nn_viz_map( fWidth, fHeight, h_snd, sWidth, sHeight, h_map, "cpu-output.bmp" );
  free( h_map );

  cutilCheckError( cutStopTimer( timer ) );
  totalTime += cutGetTimerValue( timer );

  int pSize = 2 * HALF_PATCH + 1;
  printf( "Patch size: %i x %i, Total CPU time: %10.3f msec\n", pSize, pSize, totalTime );
  printf( "         Average Iteration CPU time: %10.3f msec\n", totalTime / (float)ITERATIONS );

  /******************************************************************************/
  // GPU iterations
  totalTime = maxErr = 0;
  printf( "GPU Timer starts...\n" );

  cutilCheckError( cutResetTimer( timer ) );
  cutilCheckError( cutStartTimer( timer ) );

  // randomly allocate initial pixel map - size of original image
  mapent *curMap;
  curMap = (mapent*) malloc( fPixels * sizeof( mapent ) );
  nn_random_map( h_fst, fWidth, fHeight, h_snd, sWidth, sHeight, curMap );

  mapent *newMap;
  newMap = (mapent*) malloc( fPixels * sizeof( mapent ) );

  //cutilCheckError( cutStopTimer( timer ) );
  //totalTime += cutGetTimerValue( timer );

  // do the search
  //totalTime += nn_search_gpu( h_fst, fWidth, fHeight, h_snd, sWidth, sHeight, curMap, newMap );
  nn_search_gpu( h_fst, fWidth, fHeight, h_snd, sWidth, sHeight, curMap, newMap );

  // save final image
  nn_viz_map( fWidth, fHeight, h_snd, sWidth, sHeight, newMap, "gpu-output.bmp" );

  cutilCheckError( cutStopTimer( timer ) ); 
  totalTime += cutGetTimerValue( timer );
  //cutilSafeCall( cudaThreadSynchronize() );

  printf( "Patch size: %i x %i, Total GPU time: %10.3f msec\n", pSize, pSize, totalTime );
  printf( "         Average Iteration GPU time: %10.3f msec\n", totalTime / (float)ITERATIONS );

  free( h_fst );
  free( h_snd );
  free( newMap );
  free( curMap );

  return 0;
}

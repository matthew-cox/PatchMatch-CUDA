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
#include <common_functions.h>
#include "def.h"
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
#include "nn_gpu.hpp"
#include <math.h>
//#include <time.h>

#include "gpu_def.h"

/**********************************************************************************
 *
 * first   - first image data
 * fWidth  - width of first image
 * fHeight - height of first image
 * second  - second image data
 * sWidth  - width of second iamge
 * sHeight - height of second image
 * curMap  - existing map
 * newMap  - destination for map (assume premalloc'ed)
 *
 * Returns -
 * float of elapsed seconds
 */
float nn_search_gpu( uchar4 *first, int fWidth, int fHeight, uchar4 *second, int sWidth, int sHeight, mapent *curMap, mapent *newMap ) {

  unsigned int timer = 0;
  cutilCheckError( cutCreateTimer( &timer ) );

  cutilCheckError( cutResetTimer( timer ) );
  cutilCheckError( cutStartTimer( timer ) );

  // to copy to GPU
  int halfP = HALF_PATCH;
  int patch = 2 * HALF_PATCH + 1;

  // allocate memory and copy data
  int *R_dev;
  uchar4 *F_dev;
  mapent *curMap_dev, *newMap_dev, *temp;
  int fPixels = fWidth * fHeight;
  int sPixels = sWidth * sHeight;

  // how much RAM to allocate
  long rBytes = sizeof( int    ) * fPixels;
  long fBytes = sizeof( uchar4 ) * fPixels;
  long sBytes = sizeof( uchar4 ) * sPixels;
  long mBytes = sizeof( mapent ) * fPixels;

  if ( NNF_DEBUG ){ printf( "Allocating mem for randoms...\n" ); fflush(stdout); }
  CUDA_SAFE_CALL( cudaMalloc( (void **)&R_dev, rBytes ) );

  if ( NNF_DEBUG ){ printf( "Allocating mem for image 1...\n" ); fflush(stdout); }
  CUDA_SAFE_CALL( cudaMalloc( (void **)&F_dev, fBytes ) );

  if ( NNF_DEBUG ){ printf( "Copying image 1 data to device...\n" ); fflush(stdout); }
  CUDA_SAFE_CALL( cudaMemcpy( F_dev, first, fBytes, cudaMemcpyHostToDevice ) );

  // visualize the starting map?
  if ( NNF_VIZ ) {
    //cutilCheckError( cutStopTimer( timer ) );
    nn_viz_map( fWidth, fHeight, second, sWidth, sHeight, curMap, "gpu-iter-0.bmp" );
    //cutilCheckError( cutStartTimer( timer ) );
  }

  // FIXME: map is always the same size as the first image???
  if ( NNF_DEBUG ){ printf( "Allocating mem for maps...\n" ); fflush(stdout); }
  CUDA_SAFE_CALL( cudaMalloc( (void **)&curMap_dev, mBytes ) );
  CUDA_SAFE_CALL( cudaMalloc( (void **)&newMap_dev, mBytes ) );

  if ( NNF_DEBUG ){ printf( "Copying map data to device...\n" ); fflush(stdout); }
  CUDA_SAFE_CALL( cudaMemcpy( curMap_dev, curMap, mBytes, cudaMemcpyHostToDevice ) );
  //CUDA_SAFE_CALL( cudaMemcpy( newMap_dev, newMap, mBytes, cudaMemcpyHostToDevice ) );

  // move the params into constant memory
  if ( NNF_DEBUG ){ printf( "Copying parameters to constants...\n" ); fflush(stdout); }
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_fW", &fWidth,  sizeof(   int ) ) );
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_fH", &fHeight, sizeof(   int ) ) );
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_sW", &sWidth,  sizeof(   int ) ) );
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_sH", &sHeight, sizeof(   int ) ) );
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_hp", &halfP,   sizeof(   int ) ) );
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_ps", &patch,   sizeof(   int ) ) );

  // deploy texture cache
  CUDA_SAFE_CALL( cudaMallocArray( &dca_s, &dtl_s.channelDesc, sWidth, sHeight ) );

  // copy input data to cuda array
  CUDA_SAFE_CALL( cudaMemcpyToArray( dca_s, 0, 0, second, sBytes, cudaMemcpyHostToDevice ) );

  // configure texture cache and bind array
  dtl_s.normalized     = false;
  dtl_s.addressMode[0] = cudaAddressModeClamp;
  dtl_s.filterMode     = cudaFilterModePoint;

  CUDA_SAFE_CALL( cudaBindTextureToArray( dtl_s, dca_s ) ); //

  // set dimensions -- from lab 3
  dim3 gridDim( ceil( (float)fWidth / (float)BLOCKDIM ), ceil( (float)fHeight / (float)BLOCKDIM), 1 ); 
  dim3 blockDim( BLOCKDIM, BLOCKDIM, 1 );

  for ( int i = 1; i <= ITERATIONS; i++ ) {

    if ( NNF_DEBUG ){ printf( "Starting iteration %i...\n", i ); fflush(stdout); }

    // create array of randoms in CPU space
    int *rands;
    rands = (int*) malloc( rBytes );

    for ( int f = 0; f < fPixels; f++ ) {
      //for ( int f = 0; f < R_SEEDS; f++ ) {
      rands[f] = random();
    }

    if ( NNF_DEBUG ){ printf( "Copying randoms to device...\n" ); fflush(stdout); }
    CUDA_SAFE_CALL( cudaMemcpy( R_dev, rands, rBytes, cudaMemcpyHostToDevice ) );

    if ( NNF_DEBUG ){ printf( "Launching kernel...\n" ); fflush(stdout); }
    nn_search_kernel<<< gridDim, blockDim >>>( R_dev, F_dev, curMap_dev, newMap_dev );
    //nn_search_kernel<<< gridDim, blockDim, rBytes >>>( R_dev, F_dev, curMap_dev, newMap_dev );
    //CUDA_SAFE_CALL( cudaThreadSynchronize() );

    /* only to test the PRNG
       rand_kernel<<< gridDim, blockDim >>>( R_dev );
       CUDA_SAFE_CALL( cudaMemcpy( rands, R_dev, rBytes, cudaMemcpyDeviceToHost ) );

       for ( int f = 0; f < R_SEEDS; f++ ) {

       printf( "%i is '%i'\n", f, rands[f] );
       }*/

    // vizuaize the iterations
    if ( NNF_VIZ ) {
      char fname[64];
      sprintf( fname, "gpu-iter-%i.bmp", i );

      //cutilCheckError( cutStopTimer( timer ) );
      CUDA_SAFE_CALL( cudaMemcpy( newMap, newMap_dev, mBytes, cudaMemcpyDeviceToHost ) );
      nn_viz_map( fWidth, fHeight, second , sWidth, sHeight, newMap, fname );
      //cutilCheckError( cutStartTimer( timer ) );
    }

    // move the results - for more processing
    //CUDA_SAFE_CALL( cudaMemcpy( curMap_dev, newMap_dev, mBytes, cudaMemcpyDeviceToDevice ) );
    temp = curMap_dev;
    curMap_dev = newMap_dev;
    newMap_dev = temp;

    free( rands );
  }

  // copy results
  if ( NNF_DEBUG ){ printf( "Copying results...\n" ); fflush(stdout); }
  // FIXME: map is always the same size as the first image???
  CUDA_SAFE_CALL( cudaMemcpy( newMap, newMap_dev, mBytes, cudaMemcpyDeviceToHost ) );

  // don't time cleanup
  cutilCheckError( cutStopTimer( timer ) );

  // clean up
  if ( NNF_DEBUG ){ printf( "Freeing GPU RAM...\n" ); fflush(stdout); }
  CUDA_SAFE_CALL( cudaFree( R_dev ) );
  CUDA_SAFE_CALL( cudaFree( F_dev ) );
  CUDA_SAFE_CALL( cudaFree( curMap_dev ) );
  CUDA_SAFE_CALL( cudaFree( newMap_dev ) );
  CUDA_SAFE_CALL( cudaUnbindTexture( dtl_s ) );
  CUDA_SAFE_CALL( cudaFreeArray( dca_s ) ); //
  if ( NNF_DEBUG ){ printf( "Done.\n" ); fflush(stdout); }

  return cutGetTimerValue( timer );
}

/* ------------------------------------------------------------------------------- */
__device__ unsigned int fib_rand( int *R_dev, int f ) {

    return ( R_dev[ f - 100 ] - R_dev[ f - 37 ] ) % 1073741824L;
}

__global__ void rand_kernel( int *R_dev ) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int f = ( idy * C_fW ) + idx;
  //unsigned int r = R_dev[f];

    //r = r * 1103515245L + 12345; //(((float) (r % 65535) / 65535))
    //int cardinal = (r % 65535L) % 4; //((int)((float) (r % 65535L) / 65535L));// % 4;
    //R_dev[f] = cardinal;
    //R_dev[f] = R_dev[f] % 4;
    //R_dev[f] = fib_rand( R_de v, idx % R_SEEDS ) % 4;
		R_dev[f] = ( R_dev[ f - 100 ] - R_dev[ f - 37 ] ) % 1073741824L % 4;
}

__global__ void nn_search_kernel( int *R_dev, uchar4 *F_dev, mapent *curMap_dev, mapent *newMap_dev ) {

  // global index of the current pixel in the image
  int  idx = blockIdx.x * blockDim.x + threadIdx.x;
  int  idy = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int r;
  int fPix = C_fW * C_fH;

  // for floating point ops
  float fidx = idx;
  float fidy = idy;

  // =====================================================
  // Define shared memory
  //extern __shared__ int rands[];

  // each thread processes one pixel
  if ( idx < C_fW && idy < C_fH ) {

    int f = ( fidy * C_fW ) + fidx;

    // move randoms to shared mem
    //rands[f] = R_dev[f];
    r = R_dev[f];
    //r = rands[f];
    //__syncthreads();

    int sx, sy;
    float dist;

    // locations in the "second" image
    sx   = curMap_dev[f].x;
    sy   = curMap_dev[f].y;
    dist = curMap_dev[f].dist;

    // not at far edge
    if ( idx > 0 ) {

      int nearIdx = f - 1;

      // if the patch to the left has a better distance to its match
      // and it is not on the edge of the "second" image
      if ( ( curMap_dev[ nearIdx ].dist < dist ) && ( curMap_dev[ nearIdx ].x > 0 ) ) {

	// left patch has better dist
	sx   = curMap_dev[ nearIdx ].x - 1; 
	sy   = curMap_dev[ nearIdx ].y; 
	//dist = nn_dist_patch_gpu( F_dev, idx, idy, S_dev, sx, sy );
	dist = nn_dist_patch_gpu( F_dev, idx, idy, sx, sy );
      }
    }
      
    // check pixel above (previous row)
    if ( idy > 0 ) {

      int nearIdx = f - C_fW;

      // if the patch to the left has a better distance to its match
      // and it is not on the top edge of the "second" image
      if ( ( curMap_dev[ nearIdx ].dist < dist ) && ( curMap_dev[ nearIdx ].y > 0 ) ) {

	// up patch has better dist
	sx   = curMap_dev[ nearIdx ].x; 
	sy   = curMap_dev[ nearIdx ].y - 1; 
	//dist = nn_dist_patch_gpu( F_dev, idx, idy, S_dev, sx, sy ); 
	dist = nn_dist_patch_gpu( F_dev, idx, idy, sx, sy ); 
      }
    }

#pragma unroll 3
    // search random regions around the point
    for ( int search = 3; search > 0; search-- ) {

      // radius to use
      int radius = search * C_hp;

      int rx, ry, picked;
      rx = ry = picked = 0;
      while ( !picked ) {

	// choose north, south, east or west
	r = r * 1103515245 + 12345;
	int cardinal = (r % 65535) % 4;
	//R_dev[f] = (long)( R_dev[f - 100] - R_dev[f - 37] ) % 1073741824L;
	//int cardinal = R_dev[f] % 4;

	// store this to save operations
	r = r * 1103515245 + 12345;
	int myRand = R_dev[ (r % 65535) % fPix ];
	//R_dev[f] = (long)( R_dev[f - 100] - R_dev[f - 37] ) % 1073741824L;
	//int myRand = R_dev[f] % fPix;
	//int myRand = rands[ (r % 65535) % fPix ];

	// little switch to select a patch on the edge of radius
	if ( cardinal == 0 && sy - radius > 0 ) {
	  ry = sy - radius;
	  rx = myRand - C_hp + sx;
	  rx = min( C_sW - 1, max( 0, rx ) );
	  picked = 1;
	}
	else if ( cardinal == 1 && sx - radius > 0 ) {
	  ry = myRand - C_hp + sy;
	  ry = min( C_sH - 1, max( 0, ry ) );
	  rx = sx - radius;
	  picked = 1;
	}
	else if ( cardinal == 2 && sy + radius < C_sH ) {
	  ry = sy + radius;
	  rx = myRand - C_hp + sx;
	  rx = min( C_sW - 1, max( 0,rx) ) ;
	  picked = 1;
	}
	else if ( cardinal == 3 && sx + radius < C_sW ) {
	  ry = myRand - C_hp + sy;
	  ry = min( C_sH - 1, max( 0, ry ) );
	  rx = sx + radius;
	  picked = 1;
	}
      }

      float rdist = nn_dist_patch_gpu( F_dev, idx, idy, rx, ry );

      if ( rdist < dist ) {
	sx   = rx;
	sy   = ry;
	dist = rdist;
      } 

      newMap_dev[f].x    = sx;
      newMap_dev[f].y    = sy;
      newMap_dev[f].dist = dist;
      //	__syncthreads();
    }
    //__syncthreads();
  }
}

/*__device__ int myBadRandom( int r ) {
  r = r * 1103515245 + 12345;
  return (((float) (r % 65535) / 65535));
}*/

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
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include "def.h"
#include "nn.h"
#include "bdp.h"
#include "bmploader.h"
#include "math.h"
#include "nn_gpu.hpp"

// This is a bit of a hack to use the nn code in a different .cu file
#include "nn_gpu.cu"


/* does an initial setup of map, image data */
//__global__ void retarget_map_init_phase_1_kernel(uchar4* D_src, uchar4* D_dest, mapent* D_map) {
__global__ void retarget_map_init_phase_1_kernel(uchar4* D_src, uchar4* D_dest, mapent* D_map, int O_width, int O_height ) {

    float   xfactor = (float)C_sW / C_fW;
    float o_xfactor = (float)O_width / C_fW;
    float   yfactor = (float)C_sH / C_fH;
    float o_yfactor = (float)O_height / C_fH;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int didx = y * C_fW + x;


    if ( ( x < C_fW ) && ( y < C_fH ) ) {
       int sx = (int)floorf(xfactor*x);
       int ox = (int)floorf(o_xfactor*x);
       int sy = (int)floorf(yfactor*y);
       int oy = (int)floorf(o_yfactor*y);
       unsigned int sidx = sy * C_sW + sx;
 
       D_map[didx].x = ox;
       D_map[didx].y = oy;
       D_dest[didx].x = D_src[sidx].x;
       D_dest[didx].y = D_src[sidx].y;
       D_dest[didx].z = D_src[sidx].z;
    }
}

/*
 * src - source image data
 * s_width - width of source image
 * s_height - height of destination image
 * dest - destination image data (assume pre-malloc'd)
 * d_width - width of destination image
 * d_height - height of destination image
 * map - data map for src->dest
 */
__global__ void bidirectional_similarity_vote_kernel( uchar4* D_dest, mapent* D_map, mapent* D_revMap ) {

  
    // this value is constant for all pixels
    int num_cohere = (2*HALF_PATCH+1)*(2*HALF_PATCH+1);

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	    unsigned int idx = j*C_fW + i;
    if ( ( i < C_fW ) && ( j < C_fH ) ) {
      // sums for coherence and completeness
      float cohere_sum[3], complete_sum[3];
      cohere_sum[0] = cohere_sum[1] = cohere_sum[2] = 0;
      complete_sum[0] = complete_sum[1] = complete_sum[2] = 0;

      // number of patches that contribute for completeness
      int num_complete = 0;

      // boundries of where map will contribute to this pixel
      int patchStartX, patchEndX, patchStartY, patchEndY;
      patchStartX = i-HALF_PATCH;
      patchEndX   = i+HALF_PATCH;
      patchStartY = j-HALF_PATCH;
      patchEndY   = j+HALF_PATCH; 
 
      // loop over all parts of map that contribute
      int yoff = HALF_PATCH;
      for(int l= patchStartY; l<= patchEndY; l++) {
		
	int xoff = HALF_PATCH;
	for(int k= patchStartX; k<= patchEndX; k++) {                                       

	  // map is bound by destination image size
	  long patch_idx = max(0, min(l, C_fH-1))*C_fW + max(0, min(k,C_fW-1));
 
	  // center pixel of this patch
	  int p_x = D_map[patch_idx].x;
	  int p_y = D_map[patch_idx].y;
	  long patch_center_idx = p_y * C_sW + p_x;
	  
	  // the pixel from the source image is offset from center of this patch
	  p_x += xoff;
	  p_y += yoff;

	  // clamped at source image boundries
	  uchar4 patchPix = tex2D( dtl_s, p_x, p_y );
		    
	  // add to coherence sum
	  cohere_sum[0] += patchPix.x;
	  cohere_sum[1] += patchPix.y;
	  cohere_sum[2] += patchPix.z;
	  
	  // if the x,y stored in reverse map are the same as my coord i,j: SUCCESS!
	  if ((D_revMap[patch_center_idx].x == i) && (D_revMap[patch_center_idx].y == j)) {
	    
	    complete_sum[0] += patchPix.x;
	    complete_sum[1] += patchPix.y;
	    complete_sum[2] += patchPix.z;
	    
	    num_complete++;
	  }
	  
	  xoff--;	    
	}
	yoff--;
		    
      }

      float s_total = C_sW*C_sH;;
      float d_total = C_fW*C_fH;

      float denom = (num_complete/(s_total)) + (num_cohere/(d_total));

      complete_sum[0] /= s_total;
      complete_sum[1] /= s_total;
      complete_sum[2] /= s_total;

      cohere_sum[0] /= d_total;
      cohere_sum[1] /= d_total;
      cohere_sum[2] /= d_total;

      D_dest[idx].x = (int)floor((complete_sum[0] + cohere_sum[0])/denom);
      D_dest[idx].y = (int)floor((complete_sum[1] + cohere_sum[1])/denom);
      D_dest[idx].z = (int)floor((complete_sum[2] + cohere_sum[2])/denom);
    }
}
    
/* calculates distances for initial map */
__global__ void retarget_map_init_phase_2_kernel(uchar4* D_src, mapent* D_map) {

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    int didx = y * C_fW + x;

    D_map[didx].dist = nn_dist_patch_gpu(D_src, x, y, D_map[didx].x, D_map[didx].y);
}

bool cacheSetup = false;

void initTextureCache(uchar4* image, int width, int height) {

   if (!cacheSetup) {
     // deploy texture cache
     CUDA_SAFE_CALL( cudaMallocArray( &dca_s, &dtl_s.channelDesc, width, height ) );

     // copy input data to cuda array
     CUDA_SAFE_CALL( cudaMemcpyToArray( dca_s, 0, 0, image, width*height*sizeof(uchar4), cudaMemcpyHostToDevice ) );

     // configure texture cache and bind array
     dtl_s.normalized     = false;
     dtl_s.addressMode[0] = cudaAddressModeClamp;
     dtl_s.filterMode     = cudaFilterModePoint;

     CUDA_SAFE_CALL( cudaBindTextureToArray( dtl_s, dca_s ) ); //

     cacheSetup = true;
   }
}

void cleanupTextureCache(bool keepCache) {

  if (!keepCache) {
    CUDA_SAFE_CALL( cudaUnbindTexture( dtl_s ) );
    CUDA_SAFE_CALL( cudaFreeArray( dca_s ) );
    cacheSetup = false;
  }
}


/*
 * original - original "source" image data
 * o_width - width of original image
 * o_height - height of original image
 * src - current image
 * s_width - width of current image
 * s_height - height of current image
 * dest - destination image data (assume pre-malloc'd)
 * d_width - width of destination image
 * d_height - height of destination image
 * keepCache - whether or not to keep the cache
 */
void bidirectional_similarity_gpu_gpu(uchar4* original, int o_width, int o_height, uchar4* src, int s_width, int s_height, uchar4** dest, int d_width, int d_height, bool keepCache) {

  initTextureCache( src, o_width, o_height );

  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_fW", &d_width,  sizeof(   int ) ) );
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_fH", &d_height, sizeof(   int ) ) );
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_sW", &s_width,  sizeof(   int ) ) );
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_sH", &s_height, sizeof(   int ) ) );

  //printf("%i %i %i %i %i %i\n", o_width, o_height, s_width, s_height, d_width, d_height );

  mapent *D_map, *D_newmap, *D_revmap, *curMap, *revMap;
  uchar4 *D_src, *D_dest; 
  int *D_rands;

  curMap = (mapent *)malloc( d_width*d_height*sizeof(mapent) );

  uchar4 *newImage = (uchar4 *)malloc( d_width*d_height*sizeof(uchar4) ); 
 
  CUDA_SAFE_CALL( cudaMalloc( (void **)&D_src, s_width*s_height*sizeof(uchar4) ) );
  CUDA_SAFE_CALL( cudaMalloc( (void **)&D_dest, d_width*d_height*sizeof(uchar4) ) );

  CUDA_SAFE_CALL( cudaMalloc( (void **)&D_map, d_width*d_height*sizeof(mapent) ) );
  CUDA_SAFE_CALL( cudaMalloc( (void **)&D_newmap, d_width*d_height*sizeof(mapent)  ) );
  CUDA_SAFE_CALL( cudaMalloc( (void **)&D_revmap, o_width*o_height*sizeof(mapent)  ) );

  CUDA_SAFE_CALL( cudaMalloc( (void **)&D_rands, d_width*d_height*sizeof(int) ) );

  CUDA_SAFE_CALL( cudaMemcpy( D_src, src, s_width*s_height*sizeof(uchar4), cudaMemcpyHostToDevice ) );

  dim3 gridDim, blockDim;

  // 256 is a good starting number
  blockDim.x = 16;
  blockDim.y = 16;

  // make sure we start enough blocks
  gridDim.x = static_cast<int>( ceil( static_cast<float>(d_width) / blockDim.x ) );
  gridDim.y = static_cast<int>( ceil( static_cast<float>(d_height) / blockDim.y ) );

  //printf("%i %i %i %i\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y );
  retarget_map_init_phase_1_kernel<<< gridDim, blockDim >>>(D_src, D_dest, D_map, o_width, o_height );
  //whiteout<<< gridDim, blockDim >>>(D_dest, d_width, d_height);
    
  CUDA_SAFE_CALL( cudaMemcpy( curMap, D_map, d_width*d_height*sizeof(mapent), cudaMemcpyDeviceToHost ) );
  /*for (int x = 0; x < d_width*d_height; x++ ) {
      if (( curMap[x].x > s_width ) || (curMap[x].x < 0 )) {
	    printf("BAD X DATA at %i (%i)\n", x, curMap[x].x);
        } 
	if (( curMap[x].y > s_height ) || (curMap[x].y < 0 )) {
	    printf("BAD Y DATA at %i (%i)\n", x, curMap[x].y);
        }

    }
    printf("resetting constants...\n");*/
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_sW", &o_width,  sizeof(   int ) ) );
  CUDA_SAFE_CALL( cudaMemcpyToSymbol( (const char*)"C_sH", &o_height, sizeof(   int ) ) );
  
  //printf("Doing init2\n");
  retarget_map_init_phase_2_kernel<<< gridDim, blockDim >>>(D_dest, D_map);
    
  if (BDP_VIZ) {
    printf("Saving output\n");
    CUDA_SAFE_CALL( cudaMemcpy( newImage, D_dest, d_width*d_height*sizeof(uchar4), cudaMemcpyDeviceToHost ) );
    SaveBMPFile( newImage, d_width, d_height, "em-gpugpu-iter-0.bmp" );
  }

  for ( int emIter = 1; emIter <= ITERATIONS; emIter++ ) {
    //printf("EM iteratrion #%i\n", emIter);
    
    for ( int nnfIter = 1; nnfIter <= ITERATIONS; nnfIter++ ) {
      mapent* temp;	

      //printf("calling NNF search #%i\n", nnfIter); fflush(stdout);
    
      nn_search_kernel<<< gridDim, blockDim >>>( D_rands, D_dest, D_map, D_newmap );

      if (NNF_VIZ) {
	char fname[64];
	sprintf( fname, "em-gpugpu-iter-%i-nnf-iter-%i.bmp", emIter, nnfIter );

	CUDA_SAFE_CALL( cudaMemcpy( newImage, D_dest, d_width*d_height*sizeof(uchar4), cudaMemcpyDeviceToHost ) );
	SaveBMPFile( newImage, d_width, d_height, fname );
      }
	    
      // map new map cur map
      temp = D_map;
      D_map = D_newmap;
      D_newmap = temp;
    }
 
    //printf("copy out map\n"); fflush(stdout);
    CUDA_SAFE_CALL( cudaMemcpy( curMap, D_map, d_width*d_height*sizeof(mapent), cudaMemcpyDeviceToHost ) );

    /*for (int x = 0; x < d_width*d_height; x++ ) {
	if (( curMap[x].x > s_width ) || (curMap[x].x < 0 )) {
	    printf("BAD X DATA at %i (%i)\n", x, curMap[x].x);
        } 
	if (( curMap[x].y > s_height ) || (curMap[x].y < 0 )) {
	    printf("BAD Y DATA at %i (%i)\n", x, curMap[x].y);
        }

    }*/

    //printf("get reverse map\n"); fflush(stdout);
    gen_rev_map( curMap, d_width, d_height, &revMap, o_width, o_height );

    /*for (int x = 0; x < o_width*o_height; x++ ) {
	if (( revMap[x].x > d_width )) {
	    printf("BAD X DATA at %i (%i,%i)\n", x, revMap[x].x, d_width);
        } 
	if (( revMap[x].y > d_height )) {
	    printf("BAD Y DATA at %i (%i,%i)\n", x, revMap[x].y, d_height);
        }

    }*/

    CUDA_SAFE_CALL( cudaMemcpy( D_revmap, revMap, o_width*o_height*sizeof(mapent), cudaMemcpyHostToDevice ) );

    //printf("do bdp voting\n");
    bidirectional_similarity_vote_kernel<<< gridDim, blockDim >>>(D_dest, D_map, D_revmap);

    if (BDP_VIZ) {
      printf("Saving output\n");
      char fname[64];
      sprintf( fname, "em-gpugpu-iter-%i.bmp", emIter );

      CUDA_SAFE_CALL( cudaMemcpy( newImage, D_dest, d_width*d_height*sizeof(uchar4), cudaMemcpyDeviceToHost ) );
      SaveBMPFile( newImage, d_width, d_height, fname );
    }
  }

  //printf("copy out image\n");
  CUDA_SAFE_CALL( cudaMemcpy( newImage, D_dest, d_width*d_height*sizeof(uchar4), cudaMemcpyDeviceToHost ) );

  CUDA_SAFE_CALL( cudaFree( D_dest ) );
  CUDA_SAFE_CALL( cudaFree( D_src ) );
  CUDA_SAFE_CALL( cudaFree( D_map ) );
  CUDA_SAFE_CALL( cudaFree( D_newmap ) );
   
  cleanupTextureCache(keepCache);
 
  *dest = newImage;
}




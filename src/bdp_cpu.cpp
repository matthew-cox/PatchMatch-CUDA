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

void retarget_map_init(uchar4* src, int s_width, int s_height, uchar4** dest, int d_width, int d_height, mapent** map) {

    float xfactor = (float)s_width / d_width;
    float yfactor = (float)s_height / d_height;

    //printf( "%i/%i=%.2f %i/%i=%.2f\n", s_width, d_width, xfactor, s_height, d_height, yfactor );
    //getc(stdin);

    mapent* newMap = (mapent *)malloc( d_width * d_height * sizeof( mapent ) ); 
    uchar4* newImage = (uchar4 *)malloc( d_width * d_height * sizeof( uchar4 ) ); 

    float fx = 0;
    float fy = 0;

    for (int dy = 0; dy < d_height; dy++) {
        int sy = (int)floor(fy);

        fx = 0;
        for (int dx = 0; dx < d_width; dx++ ) {
	    int sx = (int)floor(fx);

	    //printf( "%.2f(%i),%.2f(%i) => %i, %i\n", fx,sx,fy,sy,dx,dy );

	    int didx = dy * d_width + dx;
	    int sidx = sy * s_width + sx;

            newMap[didx].x = sx;
            newMap[didx].y = sy;
	    newImage[didx] = src[sidx];

	    fx += xfactor;
	}
	
	fy += yfactor;

    }

    int idx = 0;
    for (int y = 0; y < d_height; y++) {
	for (int x = 0; x < d_width; x++) {
	    newMap[idx].dist = nn_dist_patch(src, newMap[idx].x, newMap[idx].y, s_width, s_height, newImage, x, y, d_width, d_height);
	    idx++;
 	}
    }

    *dest = newImage;
    *map = newMap;

}

void gen_rev_map(mapent *map, int m_width, int m_height, mapent **revMap, int r_height, int r_width) {

    mapent *temp = (mapent *)malloc( r_width * r_height * sizeof(mapent) );

    for ( int y = 0; y < r_height; y++ ) {
	for ( int x = 0; x < r_width; x++ ) {
	    
	    temp[y*r_width+x].x = -1;
	    temp[y*r_width+x].y = -1;
	    temp[y*r_width+x].dist = FLT_MAX;

	}
    }

    for ( int y = 0; y < m_height; y++ ) {
	for ( int x = 0; x < m_width; x++ ) {
   
 
	    int idx = y*m_width+x;
           //printf("%i,%i %i,%i\n", x, y, map[idx].x, map[idx].y ); fflush(stdout);
	    int ridx = map[idx].y*r_width+map[idx].x;

	    if (map[idx].dist < temp[ridx].dist) {
		temp[ridx].x = x;
		temp[ridx].y = y;
		temp[ridx].dist = map[idx].dist;
	    }

	}
    }

    *revMap = temp;
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
void bidirectional_similarity_vote(uchar4* src, int s_width, int s_height, uchar4* dest, int d_width, int d_height, mapent* map) {
  
    // this value is constant for all pixels
    int num_cohere = (2*HALF_PATCH+1)*(2*HALF_PATCH+1);

    // get reverse map
    mapent *revMap;
    gen_rev_map( map, d_width, d_height, &revMap, s_width, s_height );
 
    // for each pixel in result
    for (int j = 0; j < d_height; j++) {
	for (int i = 0; i < d_width; i++) {
	    unsigned int idx = j*d_width + i;

	    //printf( "starting calculations for %i,%i(%i)\n", i,j,idx ); fflush(stdout);

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
 
	    //printf( "scanning from %i,%i to %i,%i\n", patchStartX, patchStartY, patchEndX, patchEndY ); fflush(stdout);
	    
	    // loop over all parts of map that contribute
	    int yoff = HALF_PATCH;
	    for(int l= patchStartY; l<= patchEndY; l++) {
		
		int xoff = HALF_PATCH;
		for(int k= patchStartX; k<= patchEndX; k++) {                                       

		    // map is bound by destination image size
		    long patch_idx = max(0, min(l, d_height-1))*d_width + max(0, min(k,d_width-1));
 
		    // center pixel of this patch
		    int p_x = map[patch_idx].x;
		    int p_y = map[patch_idx].y;
		    long patch_center_idx = p_y * s_width + p_x;

		    // the pixel from the source image is offset from center of this patch
		    p_x += xoff;
		    p_y += yoff;

		    //printf( "patch center is %i,%i(%i<%i).\n", map[patch_idx].x, map[patch_idx].y, patch_center_idx, max_idx ); fflush(stdout);

		    // clamped at source image boundries
		    patch_idx = max(0, min(p_y, s_height-1))*s_width + max(0, min(p_x, s_width-1));
		    
		    //printf( "pixel is %i,%i(%i)\n", p_x, p_y, patch_idx ); fflush(stdout);
		    // add to coherence sum
		    cohere_sum[0] += src[patch_idx].x;
		    cohere_sum[1] += src[patch_idx].y;
		    cohere_sum[2] += src[patch_idx].z;

		    // if the x,y stored in reverse map are the same as my coord i,j: SUCCESS!
		    if ((revMap[patch_center_idx].x == i) && (revMap[patch_center_idx].y == j)) {

			//printf( "FOUND A MATCH!\n");
		    	complete_sum[0] += src[patch_idx].x;
		    	complete_sum[1] += src[patch_idx].y;
		    	complete_sum[2] += src[patch_idx].z;

			num_complete++;
		    }
			
	            //printf( "cohere is %.2f,%.2f,%.2f\n", cohere_sum[0], cohere_sum[1], cohere_sum[2] ); fflush(stdout);
		    xoff--;	    
		}
		yoff--;
		    
	    }

	    float s_total = s_width*s_height;
	    float d_total = d_width*d_height;

	    float denom = (num_complete/(s_total)) + (num_cohere/(d_total));

	    complete_sum[0] /= s_total;
	    complete_sum[1] /= s_total;
	    complete_sum[2] /= s_total;

	    cohere_sum[0] /= d_total;
	    cohere_sum[1] /= d_total;
	    cohere_sum[2] /= d_total;

	    dest[idx].x = (int)floor((complete_sum[0] + cohere_sum[0])/denom);
	    dest[idx].y = (int)floor((complete_sum[1] + cohere_sum[1])/denom);
	    dest[idx].z = (int)floor((complete_sum[2] + cohere_sum[2])/denom);
	    
            // sums and counts now fully populated for this pixel
	    //dest[idx].x = (int)floor(((complete_sum[0]/(s_width*s_height)) + (cohere_sum[0]/(d_width*d_height)))/denom);
	    //dest[idx].y = (int)floor(((complete_sum[1]/(s_width*s_height)) + (cohere_sum[1]/(d_width*d_height)))/denom);
	    //dest[idx].z = (int)floor(((complete_sum[2]/(s_width*s_height)) + (cohere_sum[2]/(d_width*d_height)))/denom);

	    //printf( "dest[%i] is %i,%i,%i\n", idx, dest[idx].x, dest[idx].y, dest[idx].z );
	    	    
	}
    }

    free(revMap);
}

/*
 * src - current image
 * s_width - width of current image
 * s_height - height of current image
 * dest - destination image data (assume pre-malloc'd)
 * d_width - width of destination image
 * d_height - height of destination image
 */
void bidirectional_similarity_EM_cpu_cpu(uchar4* src, int s_width, int s_height, uchar4* dest, int d_width, int d_height, mapent *map) {

    mapent *newMap, *oldMap;
    oldMap = (mapent*) malloc( d_width * d_height * sizeof( mapent ) );
    
    memcpy( oldMap, map, d_width * d_height * sizeof( mapent ) );

    /******************************************************************************/
    // iteration loop
    for ( int i = 1; i <= ITERATIONS; i++ ) {

    	printf( "Starting NNF iteration %i...\n", i );

        newMap = (mapent*) malloc( d_width * d_height * sizeof( mapent ) );

    	// do the search
    	nn_search( dest, d_width, d_height, src, s_width, s_height, oldMap, newMap );

    	// store the results
	free(oldMap);
    	oldMap = newMap;
    }

    // do "voting" BDS minimization
    bidirectional_similarity_vote( src, s_width, s_height, dest, d_width, d_height, oldMap );
    free(oldMap);
}

/*
 * src - current image
 * s_width - width of current image
 * s_height - height of current image
 * dest - destination image data (assume pre-malloc'd)
 * d_width - width of destination image
 * d_height - height of destination image
 */
void bidirectional_similarity_EM_cpu_gpu(uchar4* src, int s_width, int s_height, uchar4* dest, int d_width, int d_height, mapent *map) {


    /******************************************************************************/
    // nn for gpu does the interations for us 
    printf( "Starting NNF iterations...\n" );

    mapent *newMap;
    newMap = (mapent*) malloc( d_width * d_height * sizeof( mapent ) );

    // do the search
    nn_search_gpu( dest, d_width, d_height, src, s_width, s_height, map, newMap );

    // do "voting" BDS minimization
    bidirectional_similarity_vote( src, s_width, s_height, dest, d_width, d_height, newMap );

    free(newMap);
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
 */
void bidirectional_similarity_cpu_cpu(uchar4* original, int o_width, int o_height, uchar4* src, int s_width, int s_height, uchar4** dest, int d_width, int d_height) {
    mapent *map;

    printf( "Doing init\n");
    retarget_map_init(src, s_width, s_height, dest, d_width, d_height, &map);
    //bidirectional_similarity_vote( src, s_width, s_height, *dest, d_width, d_height, map );
    
    if (BDP_VIZ) {
    	SaveBMPFile( *dest, d_width, d_height, "em-cpucpu-iter-0.bmp" );
    }

    for ( int i = 1; i <= ITERATIONS; i++ ) {
	printf( "Starting EM interation %i...\n", i );	

	bidirectional_similarity_EM_cpu_cpu( original, o_width, o_height, *dest, d_width, d_height, map );
    	

    	if (BDP_VIZ) {
            char fname[64];
    	    sprintf( fname, "em-cpucpu-iter-%i.bmp", i );

    	    // vizuaize the iterations
    	    //printf( "Output map for iteration %i...\n", i );

	    SaveBMPFile( *dest, d_width, d_height, fname );
	}
    }   

    free(map);
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
 */
void bidirectional_similarity_cpu_gpu(uchar4* original, int o_width, int o_height, uchar4* src, int s_width, int s_height, uchar4** dest, int d_width, int d_height) {
    mapent *map;

    printf( "Doing init\n");
    retarget_map_init(src, s_width, s_height, dest, d_width, d_height, &map);
    //bidirectional_similarity_vote( src, s_width, s_height, *dest, d_width, d_height, map );
    
    if (BDP_VIZ) {
    	SaveBMPFile( *dest, d_width, d_height, "em-cpugpu-iter-0.bmp" );
    }

    for ( int i = 1; i <= ITERATIONS; i++ ) {
	printf( "Starting EM interation %i...\n", i );	

	bidirectional_similarity_EM_cpu_gpu( original, o_width, o_height, *dest, d_width, d_height, map );
    	
       	if (BDP_VIZ) {
            char fname[64];
    	    sprintf( fname, "em-cpugpu-iter-%i.bmp", i );

    	    // vizuaize the iterations
    	    //printf( "Output map for iteration %i...\n", i );

	    SaveBMPFile( *dest, d_width, d_height, fname );
	}
    }   

    free(map);
}




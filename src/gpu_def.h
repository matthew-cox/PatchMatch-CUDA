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

// define constant memory space for the params
__constant__ int C_sW;
__constant__ int C_sH;
__constant__ int C_fW;
__constant__ int C_fH;
__constant__ int C_hp;
__constant__ int C_ps;

#define BLOCKDIM 1 
#define MAX_KERNEL_SIZE 128

// per: http://forum.cs264.org/index.php?topic=359.msg1608#msg1608
#define rand() (r = r * 1103515245 + 12345, (((float) (r % 65535) / 65535)))

texture<uchar4, 2, cudaReadModeElementType> dtl_s;
static cudaArray* dca_s;


/*
 *
 * first - fist pixel for comparision
 * secnd - second pixel for comparison
 *
 * dist  - a float representation of SSD
 *
 */
__device__ float nn_dist_gpu( uchar4 first, uchar4 secnd ) {

  // explict casts to floats for gpu world
  float dist = sqrtf( (float)( (float)first.x - (float)secnd.x ) * ( (float)first.x - (float)secnd.x ) +
                             ( (float)first.y - (float)secnd.y ) * ( (float)first.y - (float)secnd.y ) +
                             ( (float)first.z - (float)secnd.z ) * ( (float)first.z - (float)secnd.z ) );

  /* is store faster?
  float diff1 = first.x - secnd.x;
  float diff2 = first.y - secnd.y;
  float diff3 = first.z - secnd.z;

  float dist = sqrtf( (float)__fmul_rn( diff1, diff1 ) +
                      (float)__fmul_rn( diff2, diff2 ) +
                      (float)__fmul_rn( diff3, diff3 ) );
  //float dist = sqrtf( (float)( __fmul_rz( ( (float)first.x - (float)secnd.x ), ( (float)first.x - (float)secnd.x ) ) +
                               __fmul_rz( ( (float)first.y - (float)secnd.y ), ( (float)first.y - (float)secnd.y ) ) +
                               __fmul_rz( ( (float)first.z - (float)secnd.z ), ( (float)first.z - (float)secnd.z ) ) ) ); */
  return dist;
}

//__device__ float nn_dist_patch_gpu( uchar4 *F_dev, int x1, int y1, uchar4 *S_dev, int x2, int y2 ) {
__device__ float nn_dist_patch_gpu( uchar4 *F_dev, int x1, int y1, int x2, int y2 ) {

  float dist_sum = 0;
 
  #pragma unroll 10
  for ( int j = -C_hp; j < C_hp; j++ ) {

    #pragma unroll 10
    for ( int i = -C_hp; i < C_hp; i++ ) {
      
      int ax1 = min( C_fW - 1, max( 0, x1 + i ) );
      int ay1 = min( C_fH - 1, max( 0, y1 + j ) );
			int indx1 = ay1 * C_fW + ax1;

      //int ax2 = min( C_sW - 1, max( 0, x2 + i ) );
      //int ay2 = min( C_sH - 1, max( 0, y2 + j ) );

			// fetch pixel from texture cache
      //uchar4 sP = tex2D( dtl_s, ax2+i, ay2 );
      uchar4 sP = tex2D( dtl_s, x2+i, y2+j );
      

      dist_sum += nn_dist_gpu( F_dev[indx1], sP );
    }
  }

  return dist_sum;
}

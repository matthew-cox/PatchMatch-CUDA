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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "bmploader.h"
#include "def.h"
#include "nn.h"
#include "bdp.h"

/* main for reconstruction code */
int main ( int argc, char **argv ) {
  if ( argc < 5 ) {

    printf( "Usage: retarget # input1.bmp width height\n" );
    printf( "1) Slow   - CPU code for NNF and BDS\n" );
    printf( "2) Medium - GPU NNF, CPU BDS\n" );
    printf( "3) Fast   - GPU code for NNF and BDS\n" );
    exit(1);
  }

  int pSize = 2 * HALF_PATCH + 1;

  /******************************************************************************/
  // load input BMP file(s)

  int fWidth, fHeight;
  uchar4 *h_fst, *h_snd, *h_cpuout, *h_gpuout;

  LoadBMPFile( &h_fst, &fWidth, &fHeight, argv[2] );

  int type      = atoi(argv[1]);
  int newWidth  = atoi(argv[3]);
  int newHeight = atoi(argv[4]);
 
  if ( ( newWidth == fWidth ) && ( newHeight == fHeight ) ) {
     printf("no resizing, exiting!\n");
     exit(1);
  }

  /******************************************************************************/
  // CPU iteration loop
  float totalTime    = 0;
  float maxErr       = 0;
  unsigned int timer = 0;

  switch (type) {

    case 1: 
       printf( "Starting run with CPU BDP and CPU NNF\n" );
       break;
    case 2:
       printf( "Starting run with CPU BDP and GPU NNF\n" );
       break;
    case 3:
       printf( "Starting run with GPU BDP and GPU NNF\n" );
       break;
    default:
       printf( "Invalid type! Must be 1,2, or 3\n");
       exit(1);
  }

  printf( "Starting timer...\n" );
  cutilCheckError( cutCreateTimer( &timer ) );
  cutilCheckError( cutResetTimer( timer ) );
  cutilCheckError( cutStartTimer( timer ) );

  int   diffWidth = abs(newWidth-fWidth);
  bool  growWidth = newWidth > fWidth;
  int  diffHeight = abs(newHeight-fHeight);
  bool growHeight = newHeight > fHeight;

  int curWidth  = fWidth;
  int curHeight = fHeight;

  uchar4 *curImage = h_fst;

  while ((diffWidth > 0) || (diffHeight > 0)) {

    // 5% change
    int changeWidth  = fWidth/20;
    int changeHeight = fHeight/20;

    int nextWidth    = curWidth;
    int nextHeight   = curHeight;

    changeWidth      = min(diffWidth,max(changeWidth, MIN_RETARGET));
    changeHeight     = min(diffHeight,max(changeHeight, MIN_RETARGET));

    if ( growWidth ) {
      nextWidth += changeWidth;
    }
    else {
      nextWidth -= changeWidth;
    }

    if ( growHeight ) {
      nextHeight += changeHeight;
    }
    else {
      nextHeight -= changeHeight;
    }

    //printf("resizing from %i,%i to %i,%i\n", curWidth, curHeight, nextWidth, nextHeight );

    switch (type) {

      case 1: 
        bidirectional_similarity_cpu_cpu( h_fst, fWidth, fHeight, curImage, curWidth, curHeight, &h_snd, nextWidth, nextHeight );
        break;
      case 2:
        bidirectional_similarity_cpu_gpu( h_fst, fWidth, fHeight, curImage, curWidth, curHeight, &h_snd, nextWidth, nextHeight );
        break;
      case 3:
        bidirectional_similarity_gpu_gpu( h_fst, fWidth, fHeight, curImage, curWidth, curHeight, &h_snd, nextWidth, nextHeight, ( ( nextWidth != newWidth) || ( nextHeight != newHeight ) ) );
        break;
      default:
        printf( "Invalid type! Must be 1, 2, or 3\n");
        exit( 1 );
    }

    if (curImage != h_fst) {
      free(curImage);
    }

      curImage = h_snd;
      curWidth = nextWidth;
     curHeight = nextHeight;

     char fname[64];
     sprintf( fname, "resize-%i-%i.bmp", curWidth, curHeight );
     SaveBMPFile( curImage, curWidth, curHeight, fname );

     diffWidth -= changeWidth;
    diffHeight -= changeHeight;

    //printf( "%i,%i left to go", diffWidth, diffHeight); getc(stdin);
  }

  cutilCheckError( cutStopTimer( timer ) );
  totalTime += cutGetTimerValue( timer );

  printf("saving output to 'retargeted.bmp'\n");
  SaveBMPFile(h_snd, newWidth, newHeight, "retargeted.bmp");

  printf( "Patch size: %i x %i, Total time: %.3f msec\n", pSize, pSize, totalTime );

  free( h_fst );
  free( h_snd );

  return 0;
}

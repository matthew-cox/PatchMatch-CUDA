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

#include "def.h"
#include "bmploader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "bdp.h"

/*
 *
 * fWidth  - width of first image
 * fHeight - height of first image
 * second  - second image data
 * sPixels - number of pixels in second image
 * curMap  - existing map
 * outName - output name for new bmp
 *
 */
void nn_viz_map( int fWidth, int fHeight, uchar4 *second, int sWidth, int sHeight, mapent *curMap, char *outName ) {

  int fPixels = fWidth * fHeight;

 	uchar4 *dest;
 	dest = (uchar4*) malloc( fPixels * sizeof( uchar4 ) );

  if ( NNF_VIZ_VOTE ) {

  	// use the voting algorithm for the output
  	bidirectional_similarity_vote( second, sWidth, sHeight, dest, fWidth, fHeight, curMap );
	}
	else {

  	// visualize the map - get the pixel data from the second image
  	int idx = 0;
  	for ( int y = 0; y < fHeight; y++ ) {
    	for ( int x = 0; x < fWidth; x++ ) {

      	if ( curMap[idx].x < 0 || curMap[idx].x > sWidth ) {

					printf( "Buh! Bad X posistion '%i' at %i,%i\n", curMap[idx].x, x, y );
      	}
      	else if ( curMap[idx].y < 0 || curMap[idx].y > sHeight ) {

					printf( "Buh! Bad Y posistion '%i' at %i,%i\n", curMap[idx].y, x, y );
      	}
      	else {
					int sidx = ( curMap[idx].y * sWidth ) + curMap[idx].x;
        	dest[idx] = second[sidx];
      	}
      	idx++;
    	}
  	}	 
	}

  // save output to BMP file
  printf( "nn_viz_map :: Saving data to '%s'...\n", outName );
  SaveBMPFile( dest, fWidth, fHeight, outName );
  free( dest );
}

/*
 *
 * first - fist pixel for comparision
 * secnd - second pixel for comparison
 *
 * dist  - a float representation of SSD
 *
 */
float nn_dist( uchar4 first, uchar4 secnd ) {

  float dist = sqrt( ( first.x - secnd.x ) * ( first.x - secnd.x ) +
		     ( first.y - secnd.y ) * ( first.y - secnd.y ) +
		     ( first.z - secnd.z ) * ( first.z - secnd.z ) );

  return dist;
}

float nn_dist_patch( uchar4* img1, int x1, int y1, int width1, int height1, uchar4* img2, int x2, int y2, int width2, int height2) {

  float dist_sum = 0;
 
  for ( int j = -HALF_PATCH; j < HALF_PATCH; j++ ) {
    for ( int i = -HALF_PATCH; i < HALF_PATCH; i++ ) {
      
      int ax1 = min( width1 - 1, max( 0, x1 + i ) );
      int ax2 = min( width2 - 1, max( 0, x2 + i ) );
      int ay1 = min( height1 - 1, max( 0, y1 + j ) );
      int ay2 = min( height2 - 1, max( 0, y2 + j ) );

      //printf( "diffing img1[%i,%i] and img2[%i,%i]\n", ax1, ay1, ax2, ay2 );
      //fflush(stdout);

      int indx1 = ay1 * width1 + ax1;
      int indx2 = ay2 * width2 + ax2;

      //int indx1 = (min(width1,max(0,x1+i)))+(min(height1,max(0,y1+j))*width1);
      //int indx2 = (min(width2,max(0,x2+i)))+(min(height2,max(0,y2+j))*width2);

      //printf( "diffing img1[%i] and img2[%i]\n", indx1, indx2 );
      //fflush(stdout);

      dist_sum += nn_dist( img1[indx1], img2[indx2] );
    }
  }

  return dist_sum;
}

/*
 * first   - first image data
 * fPixels - total pixels in first image
 * second  - second image data
 * sPixels - total pixels in second image
 * map     - destination for map (assume premalloc'ed)
 *
 */
void nn_random_map( uchar4 *first, int fWidth, int fHeight, uchar4 *second, int sWidth, int sHeight, mapent *map ) {

//  map = (mapent*) malloc( fPixels * sizeof( mapent ) );
  int idx = 0;
  for ( int y = 0; y < fHeight; y++ ) {
    for ( int x = 0; x < fWidth; x++ ) {
      int rx = random() % sWidth;
      int ry = random() % sHeight;
      
      map[idx].x  = rx;
      map[idx].y  = ry;
      map[idx].dist = nn_dist_patch( first, x, y, fWidth, fHeight, second, rx, ry, sWidth, sHeight );
      //printf( "Distance for %i,%i to %i,%i is '%f'\n", x, y, map[idx].x, map[idx].y , map[idx].dist );
      idx++;
    }
  }
}

/*
 * first   - first image data
 * fWidth  - width of first image
 * fHeight - height of first image
 * second  - second image data
 * sWidth  - width of second iamge
 * sHeight - height of second image
 * curMap  - existing map
 * newMap  - destination for map (assume premalloc'ed)
 *
 */
void nn_search( uchar4 *first, int fWidth, int fHeight, uchar4 *second, int sWidth, int sHeight, mapent *curMap, mapent *newMap ) {

  //int fPixels = fWidth * fHeight;
  //int sPixels = sWidth * sHeight;

  // how many rows before and after best match to look
  // goal is to check a HALD_PATCH x HALF_PATCH region around current pixel
  //int lookPix = HALF_PATCH * fWidth; // need this many rows before/after to form box
  //printf( "lookRow='%i' --- lookPix='%i'\n", HALF_PATCH, lookPix );

  int f, sx, sy;
  float dist;

  // pixels in first image loop
  for ( int fy = 0; fy < fHeight; fy++ ) {

    for ( int fx = 0; fx < fWidth; fx++ ) {

      //printf( "working on %i,%i\n", fx, fy );
      f = ( fy * fWidth ) + fx;

      // locations in the "second" image
      sx   = curMap[f].x;
      sy   = curMap[f].y;
      dist = curMap[f].dist;

      //if ( fx == 0 ) { printf( "Working on row %i...\n", fy ); }

      //if ( (int)( f / fWidth ) == (int)(fWidth / 4 ) ) { break; }
      //if ( (int)( f / fWidth ) == 2 ) { break; }

      // check the previous pixel
      //if ( f - 1 >= 0 && ( f % fWidth > 0 ) && f - 1 <= fPixels ) {

      // not at far edge
      if ( fx > 0 ) {

        int nearIdx = f - 1;

	// if the patch to the left has a better distance to its match
        // and it is not on the edge of the "second" image
        if ( ( curMap[ nearIdx ].dist < dist ) && ( curMap[ nearIdx ].x > 0 ) ) {

          //printf( "left patch has better dist...\n" );
          sx   = curMap[ nearIdx ].x - 1; 
          sy   = curMap[ nearIdx ].y; 
	  dist = nn_dist_patch( first, fx, fy, fWidth, fHeight, second, sx, sy, sWidth, sHeight );
	  //printf( "new coords are %i,%i\n", sx, sy );
	}
      }
      
      // check pixel above (previous row)
      if ( fy > 0 ) {

        int nearIdx = f - fWidth;

	// if the patch to the left has a better distance to its match
        // and it is not on the top edge of the "second" image
        if ( ( curMap[ nearIdx ].dist < dist ) && ( curMap[ nearIdx ].y > 0 ) ) {

          //printf( "up patch has better dist...\n" );
          sx   = curMap[ nearIdx ].x; 
          sy   = curMap[ nearIdx ].y - 1; 
	  dist = nn_dist_patch( first, fx, fy, fWidth, fHeight, second, sx, sy, sWidth, sHeight ); 
	  //printf( "new coords are %i,%i\n", sx, sy );
        }
      }

      // search random regions around the point
      for ( int search = 3; search > 0; search-- ) {

	// radius to use
        int radius = search * HALF_PATCH;

        int rx = 0;
        int ry = 0;        

	int picked = 0;
	while ( !picked ) {

	  // north, south, east or west
	  int cardinal = random() % 4;

	  // little switch to select a patch on the edge of radius
	  switch ( cardinal ) {

	  case 0: 
	    if ( sy - radius > 0 ) {
	      ry = sy - radius;
	      rx = ( random() % ( 2 * HALF_PATCH + 1 ) ) - HALF_PATCH + sx;
	      rx = min(sWidth-1,max(0,rx));
	      picked = 1;
	    }
	    break;

	  case 1:
	    if ( sx - radius > 0 ) {
	      ry = ( random() % ( 2 * HALF_PATCH + 1 ) ) - HALF_PATCH + sy;
	      ry = min( sHeight - 1, max( 0, ry ) );
	      rx = sx - radius;
	      picked = 1;
	    }
	    break;

	  case 2:
	    if ( sy + radius < sHeight ) {
	      ry = sy + radius;
	      rx = ( random() % ( 2 * HALF_PATCH + 1 ) ) - HALF_PATCH + sx;
	      rx = min( sWidth - 1, max( 0,rx) ) ;
	      picked = 1;
	    }
	    break;

	  case 3:
	    if ( sx + radius < sWidth ) {
	      ry = ( random() % ( 2 * HALF_PATCH + 1 ) ) - HALF_PATCH + sy;
	      ry = min( sHeight - 1, max( 0, ry ) );
	      rx = sx + radius;
	      picked = 1;
	    }
	    break;
	  }
        }

	//printf( "checking random patch %i,%i\n", rx, ry );fflush(stdout);
        float rdist = nn_dist_patch( first, fx, fy, fWidth, fHeight, second, rx, ry, sWidth, sHeight );

	//printf( "dist is %.2f", rdist );fflush(stdout);

        if ( rdist < dist ) {

	  sx   = rx;
	  sy   = ry;
	  dist = rdist;

	  //printf( "radom pixel is better\n" ); fflush(stdout);
          
        } 

        /*
      int startPos = pos;

      //printf( "startPos for %i is '%i'\n", f, startPos );
	// limit the search region
        int minPix = startPos - lookPix * search;
        if ( minPix < 0 ) { minPix = 0; }

        int maxPix = startPos + lookPix * search;
        if ( maxPix > sPixels ) { maxPix = sPixels; }

	int patchCenter = ( rand() % ( maxPix - minPix + 1 ) ) + minPix;

	int x1 = patchCenter / fWidth; // - ( search * HALF_PATCH );
	int y1 = patchCenter - ( x1 * fWidth );
        int pad = ( search * HALF_PATCH * 2 );

	//printf( "search range = '%i' -- pad = '%i'\n", search, pad );
	//printf( "%i x %i == %i\n", x1, y1, pos );

	int numChecked = 0;

	for ( int x2 = 0; x2 < fWidth; x2++ ) {

	  // limit to patch region around center pix
	  if ( x2 < ( x1 + HALF_PATCH ) && x2 > ( x1 - HALF_PATCH ) ) {

	    for ( int y2 = 0; y2 < fHeight; y2++ ) {

	      // limit to patch region around center pix
	      if ( y2 < ( y1 + HALF_PATCH ) && y2 > ( y1 - HALF_PATCH ) ) {

	         int linearPos = ( x2 * fWidth ) + y2;

		 // clamp to image boundaries
		 if ( linearPos > 0 && linearPos < sPixels ) {
	           //printf( "%i x %i == %i\n", x2, y2, linearPos );
		   numChecked++;

		  float nDist = nn_dist( first[f], second[linearPos] );

            	  if ( nDist < dist ) {

		     //printf( "Better dist (%f) found at %i\n", nDist, linearPos );
		     pos = linearPos;
		     dist = nDist;
                  }
		}
	      }
	    }
	  }*/


	}

				//printf( "Random Center = '%i' -- num checked '%i'\n", patchCenter, numChecked );
        // store the results
        //printf( "endPos for %i is '%i,%i'\n", f, sx, sy );fflush(stdout);
        newMap[f].x    = sx;
        newMap[f].y    = sy;
        newMap[f].dist = dist;
      }
   }
}

//void nn_scan( uchar4 *first, int fPix, 

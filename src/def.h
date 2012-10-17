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

#ifndef __DEF_H__
#define __DEF_H__

typedef unsigned int uint;
typedef unsigned char uchar;

#if !defined(__VECTOR_TYPES_H__)
//Isolated definition
typedef struct{
    unsigned char x, y, z, w;
} uchar4;

typedef unsigned char uchar;
#endif

// map crap
typedef struct{
    int x;
    int y;
    float dist;
} mapent;

#define ITERATIONS 5
#define HALF_PATCH 1
#define R_SEEDS 101

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) <= (b) ? (a) : (b))

// nnf debug messages
#define NNF_DEBUG 0

// save out steps of NNF iterations
#define NNF_VIZ 1

// use the BDS voting for image output
#define NNF_VIZ_VOTE 1

// save out steps of BDP iterations
#define BDP_VIZ 1

// number of pixels at which we just jump to the target
#define MIN_RETARGET 10

#endif

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

float nn_search_gpu( uchar4 *first, int fWidth, int fHeight, uchar4 *second, int sWidth, int sHeight, mapent *curMap, mapent *newMap );

__global__ void nn_search_kernel( int *rands, uchar4 *F_dev, mapent *curMap_dev, mapent *newMap_dev );

__global__ void rand_kernel( int *R_dev );

__device__ unsigned int fib_rand( int *R_dev, int f );

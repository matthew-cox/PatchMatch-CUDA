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

float nn_dist( uchar4 first, uchar4 secnd );

float nn_dist_patch( uchar4* img1, int x1, int y1, int width1, int height1, uchar4* img2, int x2, int y2, int width2, int height2 );

void nn_random_map( uchar4 *h_fst, int fWidth, int fHeight, uchar4 *h_snd, int sWidth, int sHeight, mapent *map );

void nn_search( uchar4 *first, int fWidth, int fHeight, uchar4 *second, int sWidth, int sHeight, mapent *curMap, mapent *newMap );

void nn_viz_map( int fWidth, int fHeight, uchar4 *second, int sWidth, int sHeight, mapent *map, char *outName );

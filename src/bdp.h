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

void retarget_map_init(uchar4* src, int s_width, int s_height, uchar4** dest, int d_width, int d_height, mapent** map);
void bidirectional_similarity_vote(uchar4* src, int s_width, int s_height, uchar4* dest, int d_width, int d_height, mapent* map);
void bidirectional_similarity_EM_cpu_cpu(uchar4* src, int s_width, int s_height, uchar4* dest, int d_width, int d_height, mapent *map);
void bidirectional_similarity_cpu_cpu(uchar4* original, int o_width, int o_height, uchar4* src, int s_width, int s_height, uchar4** dest, int d_width, int d_height);
void bidirectional_similarity_EM_cpu_gpu(uchar4* src, int s_width, int s_height, uchar4* dest, int d_width, int d_height, mapent *map);
void bidirectional_similarity_cpu_gpu(uchar4* original, int o_width, int o_height, uchar4* src, int s_width, int s_height, uchar4** dest, int d_width, int d_height);
void bidirectional_similarity_EM_cpu_gpu(uchar4* src, int s_width, int s_height, uchar4* dest, int d_width, int d_height, mapent *map);
void bidirectional_similarity_cpu_gpu(uchar4* original, int o_width, int o_height, uchar4* src, int s_width, int s_height, uchar4** dest, int d_width, int d_height);

void gen_rev_map(mapent *map, int m_width, int m_height, mapent **revMap, int r_height, int r_width); 

void bidirectional_similarity_gpu_gpu(uchar4* original, int o_width, int o_height, uchar4* src, int s_width, int s_height, uchar4** dest, int d_width, int d_height, bool keepCache);



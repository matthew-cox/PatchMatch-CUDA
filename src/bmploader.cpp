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

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */


#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#   pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

#pragma pack(1)

typedef struct{
    short type;
    int size;
    short reserved1;
    short reserved2;
    int offset;
} BMPHeader;

typedef struct{
    int size;
    int width;
    int height;
    short planes;
    short bitsPerPixel;
    unsigned compression;
    unsigned imageSize;
    int xPelsPerMeter;
    int yPelsPerMeter;
    int clrUsed;
    int clrImportant;
} BMPInfoHeader;



//Isolated definition
typedef struct{
    unsigned char x, y, z, w;
} uchar4;

typedef unsigned char uchar;

// making these global means that we can re-use the data when saving back out BMP files
BMPHeader hdr;
BMPInfoHeader infoHdr;


/*
//
// Read Raw image (8 bit single channel image)
//
extern "C" void LoadRawFile(uchar **dst, int &width, int &height, const char *name)
{
	FILE *fd;

    printf("Loading %s...\n", name);

    if( !(fd = fopen(name,"rb")) ){
        printf("***Raw load error: file access denied***\n");
        exit(0);
    }

	// read size of the image
	fread(&(width), sizeof(int), 1, fd);
	fread(height, sizeof(int), 1, fd);

	*dst = (uchar*)malloc(width * height);

	fread((*dst), sizeof(uchar) * width * height, 1, fd);

	fclose(fd);
}
*/

void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name)
{
    int x, y;

    FILE *fd;


    printf("Loading %s...\n", name);
    if(sizeof(uchar4) != 4){
        printf("***Bad uchar4 size***\n");
        exit(0);
    }

    if( !(fd = fopen(name,"rb")) ){
        printf("***BMP load error: file access denied***\n");
        exit(0);
    }

    fread(&hdr, sizeof(hdr), 1, fd);
    if(hdr.type != 0x4D42){
        printf("***BMP load error: bad file format***\n");
        exit(0);
    }
    fread(&infoHdr, sizeof(infoHdr), 1, fd);

    if(infoHdr.bitsPerPixel != 24){
        printf("***BMP load error: invalid color depth***\n");
        exit(0);
    }

    if(infoHdr.compression){
        printf("***BMP load error: compressed image***\n");
        exit(0);
    }

    *width  = infoHdr.width;
    *height = infoHdr.height;
    *dst    = (uchar4 *)malloc(*width * *height * 4);

    printf("BMP width: %u\n", infoHdr.width);
    printf("BMP height: %u\n", infoHdr.height);

    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr), SEEK_CUR);

    for(y = 0; y < infoHdr.height; y++){
        for(x = 0; x < infoHdr.width; x++){
            (*dst)[(y * infoHdr.width + x)].z = fgetc(fd);
            (*dst)[(y * infoHdr.width + x)].y = fgetc(fd);
            (*dst)[(y * infoHdr.width + x)].x = fgetc(fd);
        }

        for(x = 0; x < (4 - (3 * infoHdr.width) % 4) % 4; x++)
            fgetc(fd);
    }


    if(ferror(fd)){
        printf("***Unknown BMP load error.***\n");
        free(*dst);
        exit(0);
    }else
        printf("BMP file loaded successfully!\n");

    fclose(fd);
}



void SaveBMPFile(uchar4 *dst, int width, int height, const char *outputname){
    int x, y;

    FILE *fd;

    if(sizeof(uchar4) != 4){
        printf("***Bad uchar4 size***\n");
        exit(0);
    }

	// write new image
    if( !(fd = fopen(outputname,"wb")) ){
        printf("***BMP load error: file access denied***\n");
        exit(0);
    }

    infoHdr.width = width;
    infoHdr.height = height;
	fwrite(&hdr, sizeof(hdr), 1, fd);
	fwrite(&infoHdr, sizeof(infoHdr), 1, fd);



	
    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr), SEEK_CUR);
	
	// convert input to compatible to BMP
	int x_pad = (4 - (3 * infoHdr.width) % 4) % 4;
	unsigned char *out = (unsigned char *)malloc((infoHdr.width*3 + x_pad)*infoHdr.height);
	unsigned int idx = 0;
    for(y = 0; y < infoHdr.height; y++)
	{
        for(x = 0; x < infoHdr.width; x++)
		{
            out[idx++] = dst[(y * infoHdr.width + x)].z;
            out[idx++] = dst[(y * infoHdr.width + x)].y;
            out[idx++] = dst[(y * infoHdr.width + x)].x;
        }

        for(x = 0; x < x_pad; x++)
            out[idx++] = 0;
	}


	fwrite(out, (infoHdr.width*3 + x_pad)*infoHdr.height, 1, fd);

	free(out);
    fclose(fd);
}

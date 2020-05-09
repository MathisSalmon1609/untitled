#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include <math.h>

#include <IL/il.h>

int main() {

  unsigned int image;

  ilInit();

  ilGenImages(1, &image);
  ilBindImage(image);
  ilLoadImage("in.jpg");

  int width, height, bpp, format;

  width = ilGetInteger(IL_IMAGE_WIDTH);
  height = ilGetInteger(IL_IMAGE_HEIGHT);
  bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
  format = ilGetInteger(IL_IMAGE_FORMAT);

  // Récupération des données de l'image
  unsigned char* data = ilGetData();

  // Traitement de l'image
  unsigned char* out = (unsigned char*)malloc(width*height*bpp);

  unsigned int i, j, c;

  int h, m, b, res;

  struct timeval start, stop;

  gettimeofday(&start, 0);

  for(j = 1 ; j < height - 1 ; ++j) {

    for(i = 1 ; i < width - 1 ; ++i) {

      for(c = 0 ; c < 3 ; ++c) {

    h = fmax(data[((j - 1) * width + i - 1) * 3 + c], data[((j - 1) * width + i + 1) * 3 + c]);
    h = fmax(h, data[(( j - 1) * width + i) * 3 + c]);
    m = fmax(data[( j * width + i - 1) * 3 + c],  data[( j * width + i + 1) * 3 + c]);
    m = fmax(m, data[( j * width + i) * 3 + c]);
    b = fmax(data[((j + 1) * width + i - 1) * 3 + c], data[((j + 1) * width + i + 1) * 3 + c]);
    b = fmax(b, data[((j + 1) * width + i) * 3 + c]);

	res = fmax (h, m);
	res = fmax (res, b);

    res = res > 255 * 255 ? res = 255 * 255 : res;

	out[(j * width + i) * 3 + c] = res;

      }

    }

  }

  gettimeofday(&stop, 0);

  printf("time %li\n", (stop.tv_sec*1000000+stop.tv_usec) - (start.tv_sec*1000000+start.tv_usec));

  //Placement des données dans l'image

  ilSetData(out);

  // Sauvegarde de l'image

  ilEnable(IL_FILE_OVERWRITE);

  ilSaveImage("outErode.jpg");

  ilDeleteImages(1, &image);

  free(out);

}

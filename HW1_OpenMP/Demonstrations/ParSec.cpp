#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#include <omp.h>
#include <time.h>
#define RGB_COMPONENT_COLOR 255

struct PPMPixel {
    int red;
    int green;
    int blue;
};

typedef struct{
    int x, y, all;
    PPMPixel * data;
} PPMImage;

void readPPM(const char *filename, PPMImage& img){
    std::ifstream file (filename);
    if (file){
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s!="P3") {std::cout<< "error in format"<<std::endl; exit(9);}
        file >> img.x >>img.y;
        file >>rgb_comp_color;
        img.all = img.x*img.y;
        std::cout << s << std::endl;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" <<img.all;
        img.data = new PPMPixel[img.all];
        for (int i=0; i<img.all; i++){
            file >> img.data[i].red >>img.data[i].green >> img.data[i].blue;
        }

    }else{
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}

void writePPM(const char *filename, PPMImage & img){
    std::ofstream file (filename, std::ofstream::out);
    file << "P3"<<std::endl;
    file << img.x << " " << img.y << " "<< std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for(int i=0; i<img.all; i++){
        file << img.data[i].red << " " << img.data[i].green << " " << img.data[i].blue << (((i+1)%img.x ==0)? "\n" : " ");
    }
    file.close();
}

void animatePPM(PPMImage &img, PPMImage &image){
	
	image.x = img.x;
	image.y = img.y;
	image.all = image.x * image.y;

	image.data = new PPMPixel[image.all];

	double ** A = (double **)malloc(sizeof(double *)*image.x);
	double ** B = (double **)malloc(sizeof(double *)*image.x);
	double ** C = (double **)malloc(sizeof(double *)*image.x);

	for (int j = 0; j < image.x; j++)
	{
		A[j] = (double *)malloc(image.y * sizeof(double));  
		B[j] = (double *)malloc(image.y * sizeof(double));
		C[j] = (double *)malloc(image.y * sizeof(double)); 
	}

	int k = 0;
	for (int j = 0; j < image.y; j++)
	{
		for (int i = 0; i < image.x; i++)
		{
			A[i][j] = img.data[k].red;
			B[i][j] = img.data[k].blue;
			C[i][j] = img.data[k].green;
			k++;
		}
	}
	
	#pragma omp parallel shared(A, B, C, image)
	{
		#pragma omp for
		for (int j = 0; j < image.y; j++)
		{
			for (int i = 0; i < image.x; i++)
			{
				A[i][j] = A[(i + 1) % image.x][j];
				B[i][j] = B[(i + 1) % image.x][j];
				C[i][j] = C[(i + 1) % image.x][j];
			}
		}
	}
	
	k = 0;
	for (int j = 0; j < image.y; j++)
	{
		for (int i = 0; i < image.x; i++)
		{
			image.data[k].red = A[i][j];
                        image.data[k].blue = B[i][j];
                        image.data[k].green = C[i][j];
                        k++;
		}
	}
	
}

char* concat(const char *s1, const char *s2, const char *s3)
{
    char *result =(char *) malloc(strlen(s1) + strlen(s2) + strlen(s3) + 1); 

    strcpy(result, s1);
    strcat(result, s2);
    strcat(result, s3);

    return result;
}

int main(){
    PPMImage image, image_out;
    int N = 10000;
    char *filename = (char *)malloc(sizeof(char)*20);
    char ibuff[10];
    double start, end;

    readPPM("image2.ppm", image);
    omp_set_num_threads(2);

    start = omp_get_wtime();
    for (int i = 0; i < N; i++){

	sprintf(ibuff, "%d", i);
	filename = concat("image_out", (const char *) ibuff, ".ppm");
	
	animatePPM(image,image_out);
	image = image_out;

	if ((i % 200) == 0)
	{
		writePPM((const char *) filename, image_out);
	}
    }
    end = omp_get_wtime();
    printf("\nTime of calculations %.1f\n", (double)(end - start) );
    delete(image.data);  
    free(filename);
    return 0;
}

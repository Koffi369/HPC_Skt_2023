#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int apply_rules(int x, int y, int z, int *rules) {
    if (x == 1 && y == 1 && z == 1) {
        return rules[0];
    } else if (x == 1 && y == 1 && z == 0) {
        return rules[1];
    } else if (x == 1 && y == 0 && z == 1) {
        return rules[2];
    } else if (x == 1 && y == 0 && z == 0) {
        return rules[3];
    } else if (x == 0 && y == 1 && z == 1) {
        return rules[4];
    } else if (x == 0 && y == 1 && z == 0) {
        return rules[5];
    } else if (x == 0 && y == 0 && z == 1) {
        return rules[6];
    } else if (x == 0 && y == 0 && z == 0) {
        return rules[7];
    }
    return 0;
}

void random_array(int size, int * array)
{
    srand(time(NULL));
    for(int i = 0; i < size; i++) array[i] = rand() % 2;
    
}


void constant_array(int size, int * array)
{
    for(int i = 0; i < size; i++) array[i] = 0;
    array[size/2] = 1;
}



void apply_periodic_bc(int size, int *prev_array, int *rules, int *new_array, int ghost_right, int ghost_left) {
    new_array[0] = apply_rules(ghost_left, prev_array[0], prev_array[1], rules);
    new_array[size - 1] = apply_rules(prev_array[size - 2], prev_array[size-1], ghost_right, rules);
    
    for (int i = 1; i < size - 1; i++) {
      int left   = prev_array[i-1];
      int current = prev_array[i];
      int right  = prev_array[i+1];
      new_array[i] = apply_rules(left, current, right, rules);
    }
}

void apply_constant_bc(int size, int *prev_array, int *rules, int *new_array, int ghost_right, int ghost_left) {
    new_array[0] = apply_rules(1, prev_array[0], prev_array[1], rules);
    new_array[size - 1] = apply_rules(prev_array[size - 2], prev_array[size-1], 1, rules);
}


static void convert_rule_to_binary(int rule, int *converted_rule)
{
    for(int p = 0; p <= 7; p++)
    {
        if((int)(pow(2, p)) & rule)  converted_rule[abs(p - 7)] = 1;
        else converted_rule[abs(p - 7)] = 0;
    }
}

void calculate_local_size_and_place(int size, int prank, int psize, int* chunk_size, int* place)
{
    if (size % psize == 0)
    {
        chunk_size[0] = size/psize;
        place[0] = prank*chunk_size[0];
    }
    else
    {
        if (prank > size)
        {
            chunk_size[0] = 0;
            place[0] = size;
        }
        else if (prank < (size % psize))
        {
            chunk_size[0] = size/psize + 1;
            place[0] = prank*chunk_size[0];
        }
        else
        {
            chunk_size[0] = size/psize;
            place[0] = (size%psize)*(size/psize + 1) + (prank - (size % psize)) * chunk_size[0];
        }
    }
}


////////////////////////////


int main(int argc, char ** argv)
{
    
    int prank;
    int psize;
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    MPI_Request request, request2;
    
    int size = 15; 
       
    int rounds = 25;
    int rule; 
    int * converted_rule =  (int*)malloc(sizeof(int)*8);
    
    rule = atoi(argv[1]); 
    convert_rule_to_binary(rule, converted_rule);

    int * array;
    int * img_arr;
    
    if (prank == 0) {
        array = (int*)malloc(size*sizeof(int));
        // For const array
        //printf("Const array\n");
        constant_array(size, array);

        // For random array
        //printf("Random array\n");
        //random_array(size, array);

        img_arr = (int*)malloc(size*sizeof(int));
   
        printf("Constant condition\n\n"); 
        //printf("Periodic condition\n\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    int chunk_size[1];
    int place[1];

    calculate_local_size_and_place(size, prank, psize, chunk_size, place);
    
    int * chunk =  (int*)malloc(chunk_size[0] * sizeof(int));
    int process_length [psize];
    int process_place [psize];
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(&chunk_size, 1, MPI_INT, process_length, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&place, 1, MPI_INT, process_place, 1, MPI_INT, MPI_COMM_WORLD);    
    
    int strides[psize];
    strides[0] = 0;
    for (int i = 1; i < psize; i++) strides[i] = strides[i-1] + process_length[i-1];
    
    MPI_Scatterv(array, process_length, strides, MPI_INT, chunk, chunk_size[0], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    double mpi_time_start;
    if (prank == 0) mpi_time_start = MPI_Wtime();
    
    int * new_array = (int*)malloc(chunk_size[0]*sizeof(int));
    int gost_left[1], gost_right[1];
    
    
    for(int m = 0; m < rounds; m++)
    {
        MPI_Isend(&chunk[0], 1, MPI_INT, (prank-1+ psize)%psize, 0,  MPI_COMM_WORLD, &request);
        MPI_Irecv(gost_right, 1, MPI_INT, (prank+1)%psize, 0, MPI_COMM_WORLD, &request);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        MPI_Isend(&chunk[chunk_size[0]-1], 1, MPI_INT, (prank+1)%psize, 1,  MPI_COMM_WORLD, &request2);
        MPI_Irecv(gost_left, 1, MPI_INT, (prank-1+ psize)%psize, 1, MPI_COMM_WORLD, &request2);
        
        MPI_Barrier(MPI_COMM_WORLD);

        // Using Periodic BC        
        //apply_periodic_bc(chunk_size[0], chunk, converted_rule, new_array, gost_right[0], gost_left[0]);
        
        // Using constant BC
        apply_constant_bc(chunk_size[0], chunk, converted_rule, new_array, gost_right[0], gost_left[0]);

        MPI_Barrier(MPI_COMM_WORLD);
        for (int h = 0; h < size; h++) chunk[h] = new_array[h];

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gatherv(chunk, chunk_size[0], MPI_INT, img_arr, process_length, strides, MPI_INT, 0, MPI_COMM_WORLD);
        if (prank == 0)
        {
            for(int j = 0; j< size; j++)
            {
                if (img_arr[j] == 0) printf(" ");
                else if (img_arr[j] == 1) printf("*");
            }
            printf("|");
            printf("\n");
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (prank == 0)
    {
        printf("\nTotal time taken : %f seconds\n", MPI_Wtime() - mpi_time_start);
        
        free(array);
        free(img_arr);
    }

    free(chunk);
    free(converted_rule);
    free(new_array);

    MPI_Finalize();
}
















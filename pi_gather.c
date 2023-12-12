#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define CUSTOM_TAG      1
#define CUSTOM_SEED     921
#define CUSTOM_NUM_ITER 1000000000

int main(int argc, char* argv[])
{
    int local_count = 0;
    double x_val, y_val, z_val, approx_pi;
    int process_rank, process_size, i, provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    srand(process_rank * CUSTOM_SEED);

    double start_time, stop_time;
    start_time = MPI_Wtime();

    int counts[process_size];

    srand(CUSTOM_SEED * process_rank);

    // Calculate PI following a Monte Carlo method
    for (int iteration = 0; iteration < CUSTOM_NUM_ITER / process_size; iteration++)
    {
        // Generate random (X,Y) points
        x_val = (double)random() / (double)RAND_MAX;
        y_val = (double)random() / (double)RAND_MAX;
        z_val = sqrt((x_val * x_val) + (y_val * y_val));

        // Check if the point is in the unit circle
        if (z_val <= 1.0)
        {
            local_count++;
        }
    }

    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (process_rank == 0)
    {
        for (i = 0; i < process_size - 1; i++)
        {
            local_count += counts[i];
        }
        // Estimate Pi and display the result
        approx_pi = ((double)local_count / (double)CUSTOM_NUM_ITER) * 4.0;
        stop_time = MPI_Wtime();
        printf("pi=%f, t=%f\n", approx_pi, stop_time - start_time);
    }

    MPI_Finalize();
    return 0;
}

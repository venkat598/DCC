#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define CUSTOM_TAG   1
#define CUSTOM_SEED  921
#define CUSTOM_ITER  1000000000

int main(int argc, char* argv[]) {
    int local_result = 0;
    int total_result = 0; // Declare total_result to store the final result
    double x_val, y_val, z_val, approx_pi = 0.0;

    int process_rank, process_size, provided_thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided_thread_level);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    // Unique seed for each process_rank
    srand(CUSTOM_SEED * (process_rank + 1));

    double start_clock, stop_clock;
    start_clock = MPI_Wtime();

    // Calculate PI following a Monte Carlo method
    for (int iteration = 0; iteration < CUSTOM_ITER / process_size; iteration++) {
        // Generate random (X,Y) points
        x_val = (double)rand() / (double)RAND_MAX;
        y_val = (double)rand() / (double)RAND_MAX;
        z_val = sqrt((x_val * x_val) + (y_val * y_val));

        // Check if point is in unit circle
        if (z_val <= 1.0) local_result++;
    }

    // Perform MPI_Reduce to calculate the total result on process_rank 0
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&local_result, &total_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Calculate Pi on process_rank 0
    if (process_rank == 0) {
        approx_pi = ((double)total_result / (double)CUSTOM_ITER) * 4.0;
        stop_clock = MPI_Wtime();
        printf("Approximated Pi = %f, Elapsed Time = %f\n", approx_pi, stop_clock - start_clock);
    }

    MPI_Finalize();
    return 0;
}

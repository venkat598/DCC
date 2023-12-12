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

    int process_rank, process_size, provided_thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided_thread_level);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_size);

    // Ensure each process has a unique random seed
    srand(process_rank * CUSTOM_SEED);

    double start_time, stop_time, elapsed_time;
    start_time = MPI_Wtime();

    // Perform Monte Carlo simulation to estimate PI
    for (int iteration = 0; iteration < CUSTOM_NUM_ITER / process_size; iteration++)
    {
        // Generate random (X,Y) points
        x_val = (double)random() / (double)RAND_MAX;
        y_val = (double)random() / (double)RAND_MAX;
        z_val = sqrt((x_val*x_val) + (y_val*y_val));

        // Check if the point is within the unit circle
        if (z_val <= 1.0) local_count++;
    }

    // Implementing a reduction operation using a tree structure
    int depth = 0;
    while (process_size > 1) {
        int remainder = process_rank % (2 << depth);

        if (remainder) {
            // Send local count to the corresponding rank
            MPI_Send(&local_count, 1, MPI_INT, process_rank - remainder, CUSTOM_TAG, MPI_COMM_WORLD);
            break;  // Exit the loop after sending
        } else {
            // Receive and accumulate local counts
            int received_count, sender_rank = process_rank + (1 << depth);
            MPI_Recv(&received_count, 1, MPI_INT, sender_rank, CUSTOM_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_count += received_count;
            process_size /= 2;
        }
        depth++;
    }

    stop_time = MPI_Wtime();

    // Display the final result on rank 0
    if (process_rank == 0) {
        approx_pi = ((double)local_count / (double)CUSTOM_NUM_ITER) * 4.0;

        printf("Estimated value of PI: %f, Execution time: %f seconds\n", approx_pi, stop_time - start_time);
    }

    MPI_Finalize();
    return 0;
}

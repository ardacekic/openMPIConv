#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "config.h"
#include "test_input.h"
#include "test_kernel.h"
#include <string.h>

// Output tensor declaration (same logic as before)
int output_tensor[NUM_KERNELS][(INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1]
                            [(INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1];

void kernel_wise_convolution(int rank, int num_procs, int* local_kernel) {
    int kernels_per_proc = NUM_KERNELS / num_procs;
    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width  = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;

    // Local output buffer for each process
    int local_output[kernels_per_proc][output_height][output_width];
    for (int k = 0; k < kernels_per_proc; k++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                local_output[k][oh][ow] = 0;
            }
        }
    }

    // Perform convolution
    for (int k = 0; k < kernels_per_proc; k++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                for (int c = 0; c < INPUT_CHANNELS; c++) {
                    for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
                        for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                            int ih = oh * STRIDE + kh - PADDING;
                            int iw = ow * STRIDE + kw - PADDING;

                            if (ih >= 0 && iw >= 0 && ih < INPUT_HEIGHT && iw < INPUT_WIDTH) {
                                // Notice input_tensor is now guaranteed to be the same on all ranks
                                int offset = k * (KERNEL_HEIGHT * KERNEL_WIDTH * INPUT_CHANNELS)
                                       + c * (KERNEL_HEIGHT * KERNEL_WIDTH)
                                       + kh * (KERNEL_WIDTH)
                                       + kw;

                            local_output[k][oh][ow] += 
                                input_tensor[0][ih][iw][c] * local_kernel[offset];
                                //local_output[k][oh][ow] +=
                                //    input_tensor[0][ih][iw][c] * local_kernel[k][kh][kw][c];
                            }
                        }
                    }
                }
            }
        }
    }

    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather results from all processes to the root process
    int reduce_status = MPI_Gather(local_output,
                                   kernels_per_proc * output_height * output_width,
                                   MPI_INT,
                                   output_tensor,
                                   kernels_per_proc * output_height * output_width,
                                   MPI_INT,
                                   0,
                                   MPI_COMM_WORLD);

    if (reduce_status != MPI_SUCCESS) {
        fprintf(stderr, "Error in MPI_Gather on process %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, reduce_status);
    }
}

void save_output_to_file(const char *file_name, const char *file_time, double max_time) {
    FILE *file = fopen(file_name, "w");
    if (!file) {
        perror("Unable to open output file");
        exit(EXIT_FAILURE);
    }

    FILE *file2 = fopen(file_time, "w");
    if (!file2) {
        perror("Unable to open output file");
        exit(EXIT_FAILURE);
    }

    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width  = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;

    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                fprintf(file, "%d ", output_tensor[k][oh][ow]);
            }
            fprintf(file, "\n");
        }
    }

    fprintf(file2, "%.6f", max_time);

    fclose(file);
    fclose(file2);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Only rank 0 shows the total processes
    if (rank == 0) {
        printf("Number of processes: %d\n", num_procs);
    }
    int total_elements = 1 * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS;
    // Start timing
    double start = MPI_Wtime();
    // Now broadcast to all processes.
    MPI_Bcast(
        &input_tensor[0][0][0][0],  // pointer to the first element
        total_elements,             // how many ints
        MPI_INT,                    // type of data
        0,                          // root rank
        MPI_COMM_WORLD              // communicator
    );
    // -------------------------------------------------------------------------
    // Now all processes have the same input_tensor
    // -------------------------------------------------------------------------

    if (NUM_KERNELS % num_procs != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: Number of kernels (%d) must be divisible by the number "
                    "of processes (%d).\n",
                    NUM_KERNELS, num_procs);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    //Send Kernel To Processes
    int kernels_per_proc = NUM_KERNELS / num_procs;
    int num_elements = KERNEL_HEIGHT * KERNEL_HEIGHT * KERNEL_CHANNELS * kernels_per_proc;  // Elements per process

    int* local_kernel = malloc(num_elements * sizeof(int));
    int index = 0;

    if (rank == 0) {
        // Send Data
        MPI_Request *send_requests = malloc((num_procs - 1) * sizeof(MPI_Request)); // Request array for non-blocking sends
        int *all_data = (int *)malloc(NUM_KERNELS * KERNEL_HEIGHT * KERNEL_WIDTH * KERNEL_CHANNELS * sizeof(int));
        // Flattening input tensor for easier handling
        int idx_val = 0;
        for (int z = 0; z < NUM_KERNELS; z++) {
            for (int k = 0; k < INPUT_CHANNELS; k++) {
              for (int i = 0; i < KERNEL_HEIGHT; i++) {
                for (int j = 0; j < KERNEL_WIDTH; j++) {
                        all_data[idx_val] = kernels[z][i][j][k];
                        idx_val = idx_val + 1;
                        //printf("all data %d \n",kernels[z][i][j][k] );
                    }
                }
            }
        }

        for (int p = 0; p < num_procs; p++) {
            int start_idx = p * num_elements;
              if (p == 0) {
                memcpy(local_kernel, &all_data[start_idx], num_elements * sizeof(int));
            } else {
                MPI_Isend(&all_data[start_idx], num_elements, MPI_INT, p, 0, MPI_COMM_WORLD, &send_requests[p-1]);
            }
        }

        MPI_Waitall(num_procs - 1, send_requests, MPI_STATUSES_IGNORE);
        free(all_data);
        free(send_requests);

    } else {
        MPI_Recv(local_kernel, num_elements, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int idx_val = 0;
        for (int z = 0; z < kernels_per_proc; z++) {
            for (int k = 0; k < INPUT_CHANNELS; k++) {
              for (int i = 0; i < KERNEL_HEIGHT; i++) {
                for (int j = 0; j < KERNEL_WIDTH; j++) {
                        //printf("rank :%d, data %d \n",rank,local_kernel[idx_val]);
                        idx_val = idx_val + 1;
                    }
                }
            }
        }
    }

    // Perform the kernel-wise convolution
    kernel_wise_convolution(rank, num_procs,local_kernel);

    // End timing
    double end = MPI_Wtime();
    double exe_time = end - start;

    // Find the maximum execution time among all processes
    double max_time;
    MPI_Reduce(&exe_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Print final info only on rank 0
    if (rank == 0) {
        printf("Execution Time (Kernel-Wise) on rank %d: %.6f seconds\n", rank, exe_time);
        save_output_to_file("mpi_convolution_output_KRNWise.txt", "mpi_convolution_output_KRNWise_time.txt", max_time);

    }
    free(local_kernel);
    MPI_Finalize();
    return 0;
}

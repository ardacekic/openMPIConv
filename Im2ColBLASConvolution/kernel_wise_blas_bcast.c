#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h> // OpenBLAS header
#include "config.h"
#include "test_input.h"
#include "test_kernel.h"
#include <time.h>
#include <string.h>

// Output tensor declaration
int output_tensor[NUM_KERNELS][(INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1]
                            [(INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1];

// Function to perform channel-wise parallel im2col
void im2col_channel_parallel(const int input_tensor[INPUT_BATCH][INPUT_HEIGHT][INPUT_WIDTH][KERNEL_CHANNELS],
                             float *col_matrix, int batch, int rank, int num_procs) {
    int channels_per_proc = KERNEL_CHANNELS / num_procs;
    int start_channel = rank * channels_per_proc;
    int end_channel = (rank + 1) * channels_per_proc;

    // Handle any remaining channels in the last process
    if (rank == num_procs - 1) {
        end_channel = KERNEL_CHANNELS;
    }

    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;
    int col_size_per_channel = KERNEL_HEIGHT * KERNEL_WIDTH * output_height * output_width;
    float *local_col_matrix = (float *)calloc(channels_per_proc * col_size_per_channel, sizeof(float));

    int local_col_idx = 0;
    for (int c = start_channel; c < end_channel; c++) {
        for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
            for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                for (int oh = 0; oh < output_height; oh++) {
                    for (int ow = 0; ow < output_width; ow++) {
                        int ih = oh * STRIDE + kh - PADDING;
                        int iw = ow * STRIDE + kw - PADDING;
                        if (ih >= 0 && iw >= 0 && ih < INPUT_HEIGHT && iw < INPUT_WIDTH)
                            local_col_matrix[local_col_idx++] = (float)input_tensor[batch][ih][iw][c];
                        else
                            local_col_matrix[local_col_idx++] = 0.0f;  // Zero padding
                    }
                }
            }
        }
    }

    // Gather the results from all processes
    MPI_Allgather(local_col_matrix, channels_per_proc * col_size_per_channel, MPI_FLOAT, 
                  col_matrix, channels_per_proc * col_size_per_channel, MPI_FLOAT, MPI_COMM_WORLD);
    free(local_col_matrix);
}

void kernel_wise_convolution(int rank, int num_procs, float* col_matrix) {
    int kernels_per_proc = NUM_KERNELS / num_procs;
    int start_kernel = rank * kernels_per_proc;
    int end_kernel = start_kernel + kernels_per_proc;

    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;
    //int col_size = KERNEL_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * output_height * output_width;

    //float *col_matrix = (float *)malloc(col_size * sizeof(float));
    //im2col_channel_parallel(input_tensor, col_matrix, 0, rank, num_procs);
    // Broadcast the col_matrix to all processes

    float *kernel_matrix = malloc(kernels_per_proc * INPUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * sizeof(float));
    float *output_matrix = calloc(kernels_per_proc * output_height * output_width, sizeof(float));

    // Flatten relevant part of kernels
    for (int k = 0; k < kernels_per_proc; k++) {
        for (int c = 0; c < INPUT_CHANNELS; c++) {
            for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
                for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                    kernel_matrix[k * INPUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH + c * KERNEL_HEIGHT * KERNEL_WIDTH + kh * KERNEL_WIDTH + kw] = (float)kernels[start_kernel + k][kh][kw][c];
                }
            }
        }
    }

    // Perform matrix multiplication using OpenBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, kernels_per_proc, output_height * output_width, INPUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH,
                1.0f, kernel_matrix, INPUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH, col_matrix, output_height * output_width, 0.0f, output_matrix, output_height * output_width);

    // Convert output to integer and prepare for reduction
    int local_output[kernels_per_proc][output_height][output_width];
    for (int k = 0; k < kernels_per_proc; k++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                local_output[k][oh][ow] = (int)(output_matrix[k * output_height * output_width + oh * output_width + ow]);
            }
        }
    }

    // Gather results from all processes to the root process
    MPI_Gather(local_output, kernels_per_proc * output_height * output_width, MPI_INT,
               output_tensor, kernels_per_proc * output_height * output_width, MPI_INT,
               0, MPI_COMM_WORLD);

    free(kernel_matrix);
    free(output_matrix);
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
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;
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
    if (rank == 0) {
        printf("Number of processes: %d\n", num_procs);
    }
    if (NUM_KERNELS % num_procs != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of kernels (%d) must be divisible by the number of processes (%d).\n",
                    NUM_KERNELS, num_procs);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int output_height   = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width    = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;
    int col_size        = KERNEL_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * output_height * output_width;

    float *col_matrix = (float *)malloc(col_size * sizeof(float));
    im2col_channel_parallel(input_tensor, col_matrix, 0, rank, num_procs);
    
    MPI_Bcast(
        &col_matrix[0],    // pointer to the first element
        col_size,                   // how many ints
        MPI_INT,                    // type of data
        0,                          // root rank
        MPI_COMM_WORLD              // communicator
    );

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



    // Start timing
    double start = MPI_Wtime();

    // Perform kernel-wise convolution
    kernel_wise_convolution(rank, num_procs, col_matrix);

    // End timing
    double end = MPI_Wtime();
    double exe_time = end - start;

    // Find the maximum execution time across all processes
    double max_time;
    MPI_Reduce(&exe_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Execution Time (Kernel-Wise with OpenBLAS): %.6f seconds\n", max_time);
        save_output_to_file("mpi_convolution_output_BLAS_KRNLwise.txt", "mpi_convolution_output_BLAS_KRNLwise_time.txt", max_time);
    }
    free(local_kernel);
    free(col_matrix);
    MPI_Finalize();
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h> // OpenBLAS header
#include "config.h"
#include "test_input.h"
#include "test_kernel.h"
#include <time.h>

// Output tensor declaration
int output_tensor[NUM_KERNELS][(INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1]
                            [(INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1];

// Function to perform im2col
void im2col(const int input_tensor[INPUT_BATCH][INPUT_HEIGHT][INPUT_WIDTH][KERNEL_CHANNELS], float *col_matrix, int batch) {
    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;

    int col_idx = 0;
    for (int c = 0; c < INPUT_CHANNELS; c++) {
        for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
            for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                for (int oh = 0; oh < output_height; oh++) {
                    for (int ow = 0; ow < output_width; ow++) {
                        int ih = oh * STRIDE + kh - PADDING;
                        int iw = ow * STRIDE + kw - PADDING;

                        if (ih >= 0 && iw >= 0 && ih < INPUT_HEIGHT && iw < INPUT_WIDTH)
                            col_matrix[col_idx++] = (float)input_tensor[batch][ih][iw][c];
                        else
                            col_matrix[col_idx++] = 0.0f;  // Zero padding
                    }
                }
            }
        }
    }
}

void kernel_wise_convolution(int rank, int num_procs) {
    int kernels_per_proc = NUM_KERNELS / num_procs;
    int start_kernel = rank * kernels_per_proc;
    int end_kernel = start_kernel + kernels_per_proc;

    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;
    int col_size = KERNEL_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * output_height * output_width;

    // Dynamically allocate memory for input_tensor on non-root processes
    int (*local_input_tensor)[INPUT_HEIGHT][INPUT_WIDTH][KERNEL_CHANNELS] = NULL;
    if (rank != 0) {
        local_input_tensor = malloc(INPUT_BATCH * INPUT_HEIGHT * INPUT_WIDTH * KERNEL_CHANNELS * sizeof(int));
    }

    // Broadcast the input_tensor to all processes
    MPI_Bcast(rank == 0 ? input_tensor : local_input_tensor, 
              INPUT_BATCH * INPUT_HEIGHT * INPUT_WIDTH * KERNEL_CHANNELS, MPI_INT, 0, MPI_COMM_WORLD);

    float *col_matrix = calloc(col_size, sizeof(float));
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

    // Perform im2col transformation
    im2col(rank == 0 ? input_tensor : local_input_tensor, col_matrix, 0);

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

    free(col_matrix);
    free(kernel_matrix);
    free(output_matrix);

    if (rank != 0) {
        free(local_input_tensor); // Free memory allocated for non-root processes
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

    // Display the number of processes
    if (rank == 0) {
        printf("Number of processes: %d\n", num_procs);
    }

    // Check divisibility of NUM_KERNELS by num_procs
    if (NUM_KERNELS % num_procs != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of kernels (%d) must be divisible by the number of processes (%d).\n",
                    NUM_KERNELS, num_procs);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Start timing
    double start = MPI_Wtime();

    // Perform kernel-wise convolution
    kernel_wise_convolution(rank, num_procs);

    // End timing
    double end = MPI_Wtime();
    double exe_time = end - start;

    // Find the maximum execution time across all processes
    double max_time;
    MPI_Reduce(&exe_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Execution Time (Kernel-Wise with OpenBLAS): %.6f seconds\n", max_time);
        save_output_to_file("mpi_kernelwise_output.txt", "kh_wise_conv_time.txt", max_time);
    }

    MPI_Finalize();
    return 0;
}

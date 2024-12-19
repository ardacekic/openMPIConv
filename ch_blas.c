#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>  // OpenBLAS header
#include <mpi.h>
#include "config.h"
#include "test_input.h"
#include "test_kernel.h"

int output_tensor[NUM_KERNELS][(INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1][(INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1];

void im2col(const int input_tensor[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS], float *col_matrix, int start_channel, int end_channel) {
    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;
    int col_idx = 0;

    for (int c = start_channel; c < end_channel; c++) {
        for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
            for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                for (int oh = 0; oh < output_height; oh++) {
                    for (int ow = 0; ow < output_width; ow++) {
                        int ih = oh * STRIDE + kh - PADDING;
                        int iw = ow * STRIDE + kw - PADDING;
                        if (ih >= 0 && iw >= 0 && ih < INPUT_HEIGHT && iw < INPUT_WIDTH)
                            col_matrix[col_idx++] = (float)input_tensor[ih][iw][c];
                        else
                            col_matrix[col_idx++] = 0.0f;
                    }
                }
            }
        }
    }
}

void channel_wise_convolution(int rank, int num_procs) {
    int channels_per_proc = INPUT_CHANNELS / num_procs;
    int start_channel = rank * channels_per_proc;
    int end_channel = start_channel + channels_per_proc;
    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;
    int col_size = channels_per_proc * KERNEL_HEIGHT * KERNEL_WIDTH * output_height * output_width;

    float *col_matrix = calloc(col_size, sizeof(float));
    float *kernel_matrix = malloc(NUM_KERNELS * channels_per_proc * KERNEL_HEIGHT * KERNEL_WIDTH * sizeof(float));
    float *output_matrix = calloc(NUM_KERNELS * output_height * output_width, sizeof(float));

    // Flatten relevant part of kernels
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int c = 0; c < channels_per_proc; c++) {
            for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
                for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                    kernel_matrix[k * channels_per_proc * KERNEL_HEIGHT * KERNEL_WIDTH + c * KERNEL_HEIGHT * KERNEL_WIDTH + kh * KERNEL_WIDTH + kw] = (float)kernels[k][kh][kw][start_channel + c];
                }
            }
        }
    }

    // im2col for current channel slice
    im2col(input_tensor[0], col_matrix, start_channel, end_channel);

    // Matrix multiplication using OpenBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NUM_KERNELS, output_height * output_width, channels_per_proc * KERNEL_HEIGHT * KERNEL_WIDTH,
                1.0f, kernel_matrix, channels_per_proc * KERNEL_HEIGHT * KERNEL_WIDTH, col_matrix, output_height * output_width, 0.0f, output_matrix, output_height * output_width);

    // Convert output to integer and prepare for reduction
    int local_output[NUM_KERNELS][output_height][output_width];
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                local_output[k][oh][ow] = (int)(output_matrix[k * output_height * output_width + oh * output_width + ow]);
            }
        }
    }

    // Synchronize and reduce outputs
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(local_output, output_tensor, NUM_KERNELS * output_height * output_width, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    free(col_matrix);
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
        perror("Unable to open time file");
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

    if (INPUT_CHANNELS % num_procs != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: INPUT_CHANNELS must be divisible by num_procs.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    double start = MPI_Wtime();
    channel_wise_convolution(rank, num_procs);
    double end = MPI_Wtime();
    double exe_time = end - start;

    double max_time;
    MPI_Reduce(&exe_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Execution Time (Channel-Wise with OpenBLAS): %.6f seconds\n", max_time);
        save_output_to_file("mpi_convolution_output.txt", "ch_wise_conv_time.txt", max_time);
    }

    MPI_Finalize();
    return 0;
}

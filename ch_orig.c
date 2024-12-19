#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "config.h"
#include "test_input.h"
#include "test_kernel.h"


// Output tensor declaration
int output_tensor[NUM_KERNELS][(INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1]
                            [(INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1];

void channel_wise_convolution(int rank, int num_procs) {
    int channels_per_proc = INPUT_CHANNELS / num_procs;
    int start_channel = rank * channels_per_proc;
    int end_channel = start_channel + channels_per_proc;

    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;

    // Declare and initialize local_output
    int local_output[NUM_KERNELS][output_height][output_width];
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                local_output[k][oh][ow] = 0;
            }
        }
    }

    // Perform convolution
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                for (int c = start_channel; c < end_channel; c++) {
                    for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
                        for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                            int ih = oh * STRIDE + kh - PADDING;
                            int iw = ow * STRIDE + kw - PADDING;

                            if (ih >= 0 && iw >= 0 && ih < INPUT_HEIGHT && iw < INPUT_WIDTH) {
                              local_output[k][oh][ow] += (input_tensor[0][ih][iw][c]) * (kernels[k][kh][kw][c]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Reduce outputs to the root process
    int reduce_status = MPI_Reduce(local_output, output_tensor,
                                   NUM_KERNELS * output_height * output_width,
                                   MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (reduce_status != MPI_SUCCESS) {
        fprintf(stderr, "Error in MPI_Reduce on process %d.\n", rank);
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

    // Check divisibility of INPUT_CHANNELS by num_procs
    if (INPUT_CHANNELS % num_procs != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of channels (%d) must be divisible by the number of processes (%d).\n",
                    INPUT_CHANNELS, num_procs);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Start timing
    double start = MPI_Wtime();

    // Perform channel-wise convolution
    channel_wise_convolution(rank, num_procs);

    // End timing
    double end = MPI_Wtime();
    double exe_time = end - start;

    // Find the maximum execution time across all processes
    double max_time;
    MPI_Reduce(&exe_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Execution Time (Channel-Wise) in C %d rank: %.6f seconds\n",rank, exe_time);
        save_output_to_file("mpi_convolution_output.txt", "ch_wise_conv_time.txt", max_time);
    }

    MPI_Finalize();
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "config.h"
#include "test_input.h"
#include "test_kernel.h"
#include <string.h>

// Assuming the dimensions for input and kernel are defined in 'config.h'

// Declare the output tensor globally
int output_tensor[NUM_KERNELS][(INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1]
                            [(INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1];

// Function to perform channel-wise convolution
void channel_wise_convolution(int rank, int num_procs, int *local_input) {
    int channels_per_proc = INPUT_CHANNELS / num_procs;
    int num_elements = INPUT_HEIGHT * INPUT_WIDTH * channels_per_proc;  // Elements per process
    int start_channel = rank * channels_per_proc;
    int end_channel = start_channel + channels_per_proc;

    int local_input_size = channels_per_proc * INPUT_HEIGHT * INPUT_WIDTH;
    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;

    // Allocate memory for local input and output
    int (*local_output)[output_height][output_width] = calloc(NUM_KERNELS, sizeof(*local_output));

    // Scatter the input tensor to all processes
    //MPI_Barrier(MPI_COMM_WORLD);
    // Perform convolution on local input
    for (int k = 0; k < NUM_KERNELS; k++) {
        int flat_idx = 0;
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                for (int c = start_channel; c < end_channel; c++) {
                    for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
                        for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                            int ih = oh * STRIDE + kh - PADDING;
                            int iw = ow * STRIDE + kw - PADDING;
                            if (ih >= 0 && iw >= 0 && ih < INPUT_HEIGHT && iw < INPUT_WIDTH) {
                                flat_idx = ((c - start_channel) * INPUT_HEIGHT + ih) * INPUT_WIDTH + iw;
                                local_output[k][oh][ow] += local_input[flat_idx] * kernels[k][kh][kw][c];                        
                            }
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Reduce all local outputs to the root process
    MPI_Reduce(local_output, output_tensor,
               NUM_KERNELS * output_height * output_width, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Free allocated memory
    free(local_output);
}

// Function to save output to file
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

// Main function
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
            fprintf(stderr, "Error: Number of channels (%d) must be divisible by the number of processes (%d).\n",
                    INPUT_CHANNELS, num_procs);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    double start = MPI_Wtime();

    int channels_per_proc = INPUT_CHANNELS / num_procs;
    int num_elements = INPUT_HEIGHT * INPUT_WIDTH * channels_per_proc;  // Elements per process

    int* local_input = malloc(num_elements * sizeof(int));
    int index = 0;
    if (rank == 0) {
        // Send Data
        MPI_Request *send_requests = malloc((num_procs - 1) * sizeof(MPI_Request)); // Request array for non-blocking sends
        int *all_data = (int *)malloc(INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * sizeof(int));
        // Flattening input tensor for easier handling
        int idx_val = 0;
        for (int k = 0; k < INPUT_CHANNELS; k++) {
          for (int i = 0; i < INPUT_HEIGHT; i++) {
            for (int j = 0; j < INPUT_WIDTH; j++) {
                    all_data[idx_val] = input_tensor[0][i][j][k];
                    idx_val = idx_val + 1;
                }
            }
        }

        for (int p = 0; p < num_procs; p++) {
            int start_idx = p * num_elements;
              if (p == 0) {
                memcpy(local_input, &all_data[start_idx], num_elements * sizeof(int));
            } else {
                MPI_Isend(&all_data[start_idx], num_elements, MPI_INT, p, 0, MPI_COMM_WORLD, &send_requests[p-1]);
            }
        }

        MPI_Waitall(num_procs - 1, send_requests, MPI_STATUSES_IGNORE);
        free(all_data);
    }else{
             int idx_val = 0;
    for (int k = 0; k < channels_per_proc; k++) {
      for (int i = 0; i < INPUT_HEIGHT; i++) {
        for (int j = 0; j < INPUT_WIDTH; j++) {
                //printf("Rank %d val %d :\n ", rank, local_input[idx_val]);
                idx_val = idx_val + 1;
            }
        }
    }
      MPI_Recv(local_input, num_elements, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    


    channel_wise_convolution(rank, num_procs,local_input);

    double end = MPI_Wtime();
    double exe_time = end - start;

    double max_time;
    MPI_Reduce(&exe_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Execution Time (Channel-Wise Scatter): %.6f seconds\n", max_time);
        save_output_to_file("mpi_convolution_output_CHWise.txt", "mpi_convolution_output_CHWise_time.txt", max_time);
    }
    free(local_input);
    MPI_Finalize();
    return 0;
}

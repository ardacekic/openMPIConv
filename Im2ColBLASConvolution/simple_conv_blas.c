#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>  // OpenBLAS header
#include <time.h>
#include "config.h"
#include "test_input.h"
#include "test_kernel.h"

int output_tensor[INPUT_BATCH][NUM_KERNELS][(INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1][(INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1];

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

// Updated convolve function using im2col and OpenBLAS
void convolve() {
    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;
    int col_size = KERNEL_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * output_height * output_width;

    // Flatten kernels into a 2D matrix
    float *kernel_matrix = malloc(NUM_KERNELS * KERNEL_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * sizeof(float));
    for (int k = 0; k < NUM_KERNELS; k++) {
        for (int c = 0; c < KERNEL_CHANNELS; c++) {
            for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
                for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                    kernel_matrix[k * KERNEL_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH + c * KERNEL_HEIGHT * KERNEL_WIDTH + kh * KERNEL_WIDTH + kw] = (float)kernels[k][kh][kw][c];
                }
            }
        }
    }

    for (int b = 0; b < INPUT_BATCH; b++) {
        float *col_matrix = calloc(col_size, sizeof(float));
        im2col(input_tensor, col_matrix, b);

        float *output_matrix = calloc(NUM_KERNELS * output_height * output_width, sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NUM_KERNELS, output_height * output_width, KERNEL_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH,
                    1.0f, kernel_matrix, KERNEL_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH, col_matrix, output_height * output_width, 0.0f, output_matrix, output_height * output_width);

        for (int k = 0; k < NUM_KERNELS; k++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    output_tensor[b][k][oh][ow] = (int)(output_matrix[k * output_height * output_width + oh * output_width + ow]);  // Round to nearest int
                }
            }
        }

        free(col_matrix);
        free(output_matrix);
    }
    free(kernel_matrix);
}

void save_output_to_file(const char *file_name, const char *time_file, double exeTime) {
    FILE *file = fopen(file_name, "w");
    if (!file) {
        perror("Unable to open output file");
        exit(EXIT_FAILURE);
    }

    FILE *file2 = fopen(time_file, "w");
    if (!file2) {
        perror("Unable to open time file");
        exit(EXIT_FAILURE);
    }

    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;

    for (int b = 0; b < INPUT_BATCH; b++) {
        for (int k = 0; k < NUM_KERNELS; k++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    fprintf(file, "%d ", output_tensor[b][k][oh][ow]);
                }
                fprintf(file, "\n");
            }
        }
    }
    fprintf(file2, "%f", exeTime);
    fclose(file);
    fclose(file2);
}

int main() {
    clock_t start = clock();
    convolve();
    clock_t end = clock();

    double exeTime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution Time with OpenBLAS: %.6f seconds\n", exeTime);
    save_output_to_file("mpi_convolution_output_BLAS_Simple.txt", "mpi_convolution_output_Simple_time.txt", exeTime);


    return 0;
}

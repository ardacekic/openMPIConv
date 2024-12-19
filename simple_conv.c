#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "test_input.h"
#include "test_kernel.h"
#include <time.h>

int output_tensor[INPUT_BATCH][NUM_KERNELS][(INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1][(INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1];

void convolve() {
    int output_height = (INPUT_HEIGHT - KERNEL_HEIGHT + 2 * PADDING) / STRIDE + 1;
    int output_width = (INPUT_WIDTH - KERNEL_WIDTH + 2 * PADDING) / STRIDE + 1;

    for (int b = 0; b < INPUT_BATCH; b++) {
        for (int k = 0; k < NUM_KERNELS; k++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    int sum = 0.0;

                    for (int c = 0; c < KERNEL_CHANNELS; c++) {
                        for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
                            for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                                int ih = oh * STRIDE + kh - PADDING;
                                int iw = ow * STRIDE + kw - PADDING;

                                if (ih >= 0 && iw >= 0 && ih < INPUT_HEIGHT && iw < INPUT_WIDTH) {
                                    sum += input_tensor[b][ih][iw][c] * kernels[k][kh][kw][c];
                                }
                            }
                        }
                    }

                    output_tensor[b][k][oh][ow] = sum;
                }
            }
        }
    }
}

void save_output_to_file(const char *file_name,const char *time_file, double exeTime ) {
    FILE *file = fopen(file_name, "w");
    if (!file) {
        perror("Unable to open output file");
        exit(EXIT_FAILURE);
    }

    FILE *file2 = fopen(time_file, "w");
    if (!file2) {
        perror("Unable to open output file");
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
    fprintf(file2, "%f",exeTime);
    fclose(file);
    fclose(file2);
}

int main() {
    clock_t start = clock();
    convolve();
    clock_t end = clock();

    double exeTime = (double)(end-start) / CLOCKS_PER_SEC;
    printf("Execution Time in C: : %.6f seconds \n",exeTime);
    save_output_to_file("c_convolution_output.txt","simple_conv_time.txt",exeTime);
    return 0;
}

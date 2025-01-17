#ifndef CONFIG_H
#define CONFIG_H

// Input Tensor Configuration
#define INPUT_BATCH 1
#define INPUT_HEIGHT 64
#define INPUT_WIDTH 64
#define INPUT_CHANNELS 1024

// Kernel Configuration
#define NUM_KERNELS 128
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3
#define KERNEL_CHANNELS 1024

// Convolution Parameters
#define STRIDE 1
#define PADDING 0

#define NUMCORE 64
#endif // CONFIG_H

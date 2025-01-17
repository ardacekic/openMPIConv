import numpy as np
import torch
import torch.nn.functional as F
import re
import time

# Function to parse C header file (config.h) for parameter values
def parse_config(file_path):
    config = {}
    with open(file_path, "r") as f:
        for line in f:
            match = re.match(r"#define\s+(\w+)\s+(\d+)", line)
            if match:
                key, value = match.groups()
                config[key] = int(value)
    return config

# Load configuration from config.h
config = parse_config("config.h")

INPUT_BATCH = config["INPUT_BATCH"]
INPUT_HEIGHT = config["INPUT_HEIGHT"]
INPUT_WIDTH = config["INPUT_WIDTH"]
INPUT_CHANNELS = config["INPUT_CHANNELS"]
NUM_KERNELS = config["NUM_KERNELS"]
KERNEL_HEIGHT = config["KERNEL_HEIGHT"]
KERNEL_WIDTH = config["KERNEL_WIDTH"]
KERNEL_CHANNELS = config["KERNEL_CHANNELS"]
STRIDE = config.get("STRIDE", 1)
PADDING = config.get("PADDING", 0)
NUMCORE = config.get("NUMCORE",1)

def load_tensor_from_npy(file_name):
    """
    Loads a tensor from a NumPy binary file.

    :param file_name: Name of the .npy file containing the tensor
    :return: Loaded tensor as a NumPy array
    """
    return np.load(file_name)

def run_convolution(NUMCORE=NUMCORE,input_file="test_input.npy", kernel_file="test_kernel.npy", output_file="convolution_output.txt"):
    """
    Performs a convolution operation using PyTorch with the given input tensor and kernels.

    :param input_file: Path to the input tensor file
    :param kernel_file: Path to the kernel tensor file
    :param output_file: Path to save the rounded output tensor
    """
    # Load input tensor and kernels
    input_tensor = load_tensor_from_npy(input_file)
    kernels = load_tensor_from_npy(kernel_file)

    # Convert to PyTorch tensors
    input_tensor_torch = torch.tensor(input_tensor, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
    kernels_torch = torch.tensor(kernels, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
    print(f"NUMCORE : {NUMCORE}")
    torch.set_num_threads(NUMCORE)

    start_time = time.time()
    # Perform convolution
    output = F.conv2d(
        input_tensor_torch,
        kernels_torch,
        stride=STRIDE,
        padding=PADDING
    )


    # Convert output to NumPy and round to 2 decimal places with fixed-point formatting
    output_np = output.detach().numpy()
    output_rounded = np.round(output_np, 2)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution Time in Python: {execution_time:.6f} seconds")
    with open("python_time_out.txt", "w") as f:
      f.write(f"{execution_time:.6f}")
      
    # Save only the output values to a text file
    with open(output_file, "w") as f:
        for batch in range(output_rounded.shape[0]):
            for kernel in range(output_rounded.shape[1]):
                for row in output_rounded[batch, kernel]:
                    f.write(" ".join(f"{value:.0f}" for value in row) + "\n")

    return output_rounded

# Run convolution on generated input and kernel data
run_convolution()


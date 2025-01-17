import numpy as np
import re

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

def generate_input_tensor(output_file_h="test_input.h", output_file_npy="test_input.npy"):
    """
    Generates a random input tensor based on the configuration file and saves it to .h and .npy files.
    """
    tensor = np.random.randint(low=0, high=256, size=(INPUT_BATCH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)).astype(np.int8)

    # Save to .h file
    with open(output_file_h, "w") as f:
        f.write("// Auto-generated input tensor\n")
        f.write(f"#define INPUT_BATCH {INPUT_BATCH}\n")
        f.write(f"#define INPUT_HEIGHT {INPUT_HEIGHT}\n")
        f.write(f"#define INPUT_WIDTH {INPUT_WIDTH}\n")
        f.write(f"#define INPUT_CHANNELS {INPUT_CHANNELS}\n")
        f.write("int input_tensor[INPUT_BATCH][INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS] = {\n")

        for b in range(INPUT_BATCH):
            f.write(" {\n")
            for h in range(INPUT_HEIGHT):
                f.write("  {\n")
                for w in range(INPUT_WIDTH):
                    f.write("   { " + ", ".join(map(str, tensor[b, h, w, :])) + " },\n")
                f.write("  },\n")
            f.write(" },\n")
        f.write("};\n")

    # Save to .npy file
    np.save(output_file_npy, tensor)

def generate_kernel_data(output_file_h="test_kernel.h", output_file_npy="test_kernel.npy"):
    """
    Generates random kernel data based on the configuration file and saves it to .h and .npy files.
    """
    kernels = np.random.randint(low=0, high=256, size=(NUM_KERNELS, KERNEL_HEIGHT, KERNEL_WIDTH, KERNEL_CHANNELS)).astype(np.int8)

    # Save to .h file
    with open(output_file_h, "w") as f:
        f.write("// Auto-generated kernel data\n")
        f.write(f"#define NUM_KERNELS {NUM_KERNELS}\n")
        f.write(f"#define KERNEL_HEIGHT {KERNEL_HEIGHT}\n")
        f.write(f"#define KERNEL_WIDTH {KERNEL_WIDTH}\n")
        f.write(f"#define KERNEL_CHANNELS {KERNEL_CHANNELS}\n")
        f.write("int kernels[NUM_KERNELS][KERNEL_HEIGHT][KERNEL_WIDTH][KERNEL_CHANNELS] = {\n")

        for n in range(NUM_KERNELS):
            f.write(" {\n")
            for h in range(KERNEL_HEIGHT):
                f.write("  {\n")
                for w in range(KERNEL_WIDTH):
                    f.write("   { " + ", ".join(map(str, kernels[n, h, w, :])) + " },\n")
                f.write("  },\n")
            f.write(" },\n")
        f.write("};\n")

    # Save to .npy file
    np.save(output_file_npy, kernels)

# Generate both input tensor and kernel data
generate_input_tensor()
generate_kernel_data()

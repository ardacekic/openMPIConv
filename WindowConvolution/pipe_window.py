import os
import matplotlib.pyplot as plt
import numpy as np

channel_sizes = [2048]
kernel_sizes = [512]
np_values = [2,4,8,16,32,64]  # Number of processes to test

def update_config(input_channels, kernel_size, np_value):
    config_template = f"""#ifndef CONFIG_H
#define CONFIG_H

// Input Tensor Configuration
#define INPUT_BATCH 1
#define INPUT_HEIGHT 64
#define INPUT_WIDTH 64 
#define INPUT_CHANNELS {input_channels}

// Kernel Configuration
#define NUM_KERNELS {kernel_size}
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3
#define KERNEL_CHANNELS {input_channels}

// Convolution Parameters
#define STRIDE 1
#define PADDING 0

#define NUMCORE {np_value}
#endif // CONFIG_H
"""
    with open("config.h", "w") as f:
        f.write(config_template)

    print(f"Updated config.h with INPUT_CHANNELS={input_channels}, KERNEL_SIZE={kernel_size}, np_value = {np_value}")

def collect_execution_times(filenames):
    execution_times = []

    for file in filenames:
        try:
            with open(file, "r") as f:
                content = f.read().strip()
                execution_times.append(float(content))  # Convert to float and store
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            execution_times.append(0.0)  # Append 0.0 if there's an issue

    return execution_times

def plot_speedup_simple_vs_chwise(channel_sizes, kernel_sizes, np_values, ch_kernel_collections):
    execution_times = np.load('ch_kernel_collections.npy')

    plt.figure(figsize=(12, 8))

    # Preparing labels and data for plotting
    x_labels = [f"np={np_value}" for np_value in np_values]
    x = np.arange(len(x_labels))

    # Speedups for each configuration
    speedup_ch_wise = []
    for i in range(len(np_values)):
        simple_time = execution_times[i, 1]
        speedup_ch_wise.append(simple_time / execution_times[i, 2])

    # Plotting the speedups
    plt.plot(x, speedup_ch_wise, label='Speedup vs. Channel-Wise', marker='^', color='blue')

    plt.title("Speedup: Simple Convolution vs. Channel-Wise Convolution")
    plt.xlabel("Number of Processes (np)")
    plt.ylabel("Speedup (Simple Time / Channel-Wise Time)")
    plt.xticks(x, x_labels, ha="center")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    filename = "window_multiplication_wo_blas_plot_speedup_simple_vs_chwise.png"
    plt.savefig(filename)
    print(f"Combined plot saved as {filename}")
    plt.tight_layout()
    plt.show()

def plot_speedup_simple_vs_krnwise(channel_sizes, kernel_sizes, np_values, ch_kernel_collections):
    execution_times = np.load('ch_kernel_collections.npy')

    plt.figure(figsize=(12, 8))

    # Preparing labels and data for plotting
    x_labels = [f"np={np_value}" for np_value in np_values]
    x = np.arange(len(x_labels))

    # Speedups for each configuration
    speedup_kh_wise = []
    for i in range(len(np_values)):
        simple_time = execution_times[i, 1]
        speedup_kh_wise.append(simple_time / execution_times[i, 3])

    # Plotting the speedups
    plt.plot(x, speedup_kh_wise, label='Speedup vs. Kernel-Wise', marker='d', color='green')

    plt.title("Speedup: Simple Convolution vs. Kernel-Wise Convolution")
    plt.xlabel("Number of Processes (np)")
    plt.ylabel("Speedup (Simple Time / Kernel-Wise Time)")
    plt.xticks(x, x_labels, ha="center")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    filename = "window_multiplication_wo_blas_plot_speedup_simple_vs_krnwise.png"
    plt.savefig(filename)
    print(f"Combined plot saved as {filename}")
    plt.tight_layout()
    plt.show()

def plot_speedup_comparison(channel_sizes, kernel_sizes, np_values, ch_kernel_collections):
    execution_times = np.load('ch_kernel_collections.npy')

    plt.figure(figsize=(20, 12))

    # Preparing labels and data for plotting
    x_labels = [f"np={np_value}" for np_value in np_values]
    x = np.arange(len(x_labels))

    # Speedups for each configuration
    speedup_ch_wise = []
    speedup_kh_wise = []
    for i in range(len(np_values)):
        simple_time = execution_times[i, 1]
        speedup_ch_wise.append(simple_time / execution_times[i, 2])
        speedup_kh_wise.append(simple_time / execution_times[i, 3])

    # Plotting the speedups
    plt.plot(x, speedup_ch_wise, label='Channel-Wise Speedup', marker='^')
    plt.plot(x, speedup_kh_wise, label='Kernel-Wise Speedup', marker='d')

    plt.title("Speedup Comparison (Simple vs Channel/Kernel Wise)")
    plt.xlabel("Number of Processes (np)")
    plt.ylabel("Speedup (Simple Time / Conv Time)")
    plt.xticks(x, x_labels, ha="center")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    filename = "window_multiplication_wo_blas_plot_speedup_comparison.png"
    plt.savefig(filename)
    print(f"Combined plot saved as {filename}")
    plt.tight_layout()
    plt.show()

def plot_all_execution_times(channel_sizes, kernel_sizes, np_values, ch_kernel_collections):
    execution_times = np.load('ch_kernel_collections.npy')
      #np.array(ch_kernel_collections)

    plt.figure(figsize=(20, 12))  # Further increase in figure size

    x_labels = [
        f"{channel},{kernel},{np_value}"
        for channel in channel_sizes
        for kernel in kernel_sizes
        for np_value in np_values
    ]
    x = np.arange(len(x_labels))

    time_python = execution_times[:, 0]
    time_simple_conv = execution_times[:, 1]
    time_ch_wise = execution_times[:, 2]
    time_kh_wise = execution_times[:, 3]

    plt.plot(x, time_python, label='Python Time', marker='o')
    plt.plot(x, time_simple_conv, label='Simple Conv Time', marker='s')
    plt.plot(x, time_ch_wise, label='Channel-Wise Conv Time', marker='^')
    plt.plot(x, time_kh_wise, label='Kernel-Wise Conv Time', marker='d')

    plt.title("Execution Times for All Configurations")
    plt.xlabel("Configurations (Channel, Kernel, np)")
    plt.ylabel("Execution Time (s)")
    plt.xticks(x, x_labels, rotation=90, ha="center")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    filename = "window_multiplication_wo_blas.png"
    plt.savefig(filename)
    print(f"Combined plot saved as {filename}")

    plt.tight_layout()
    plt.show()

def plot_all_execution_times_nosimple(channel_sizes, kernel_sizes, np_values, ch_kernel_collections):
    execution_times = np.load('ch_kernel_collections.npy')
      #np.array(ch_kernel_collections)

    plt.figure(figsize=(20, 12))  # Further increase in figure size

    x_labels = [
        f"{channel},{kernel},{np_value}"
        for channel in channel_sizes
        for kernel in kernel_sizes
        for np_value in np_values
    ]
    x = np.arange(len(x_labels))

    time_python = execution_times[:, 0]
    #time_simple_conv = execution_times[:, 1]
    time_ch_wise = execution_times[:, 2]
    time_kh_wise = execution_times[:, 3]

    plt.plot(x, time_python, label='Python Time', marker='o')
    #plt.plot(x, time_simple_conv, label='Simple Conv Time', marker='s')
    plt.plot(x, time_ch_wise, label='Channel-Wise Conv Time', marker='^')
    plt.plot(x, time_kh_wise, label='Kernel-Wise Conv Time', marker='d')

    plt.title("Execution Times for All Configurations")
    plt.xlabel("Configurations (Channel, Kernel, np)")
    plt.ylabel("Execution Time (s)")
    plt.xticks(x, x_labels, rotation=90, ha="center")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    filename = "window_multiplication_wo_blas_nosimple.png"
    plt.savefig(filename)
    print(f"Combined plot saved as {filename}")

    plt.tight_layout()
    plt.show()

only_once = 1
ch_kernel_collections = []
for channels in channel_sizes:
    for kernel_size in kernel_sizes:
        for np_value in np_values:
            itt = 1
            print("**** STARTS ****")
            print("Number of TOTAL CPUs:", os.cpu_count())
            print(f"Now using Number of CPU: {np_value}")
            print(f"Channel {channels}, Kernel {kernel_size}, np={np_value}")
            update_config(channels, kernel_size, np_value)

            os.system("python3 data_generate.py")

            filenames = [
                "python_time_out.txt",
                "mpi_convolution_output_Simple_time.txt",
                "mpi_convolution_output_CHWise_time.txt",
                "mpi_convolution_output_KRNWise_time.txt"
            ]

            cumulative_execution = np.zeros(len(filenames))  # Initialize with zeros
            for _ in range(100):
                print(f"Iteration: {itt}")
                itt += 1

                print(f"Run Python Stage Started:")
                os.system("python3 run_conv.py")

                if only_once == 1:
                    print(f"Compile Simple C model Started:")
                    os.system("gcc -O1 -std=c99 -o conv.o simple_conv.c -lm ")
                    print(f"Run Simple C model Started:")
                    os.system("./conv.o")
                    os.system("python3 compare_Window_Simple.py")
                    only_once = 0

                print(f"Compile CHwise model Started:")
                os.system(f"mpicc -O2 -o ch_conv_orig.o mpi_conv_scatter.c")
                print(f"Run CHwise model Started:")
                os.system(f"mpirun --allow-run-as-root -np {np_value} ./ch_conv_orig.o")
                os.system("python3 compare_Window_CH.py")

                print(f"Compile KRNwise model Started:")
                os.system(f"mpicc -O2 -o kernel_conv.o kernel_wise_bcast.c")
                print(f"Run KRNwise model Started:")
                os.system(f"mpirun --allow-run-as-root -np {np_value} ./kernel_conv.o")
                os.system("python3 compare_Window_KRN.py")

                current_execution = np.array(collect_execution_times(filenames))  # Convert to numpy array
                print(f"Current: {current_execution}")
                cumulative_execution += current_execution  # Element-wise summation
                print(f"Cumulative: {cumulative_execution}")

                if itt > 1:  # Exit after 3 iterations for demonstration
                    average_execution = cumulative_execution / (itt-1)  # Compute the average
                    print(f"Averaged Execution Times: {average_execution}")
                    ch_kernel_collections.append(average_execution.tolist())
                    print("**** ENDS ****")
                    break

print("Final execution times array:", ch_kernel_collections)
np.save('ch_kernel_collections.npy', ch_kernel_collections)

# Plot the collected execution times for all configurations
plot_all_execution_times(channel_sizes, kernel_sizes, np_values, ch_kernel_collections)
plot_all_execution_times_nosimple(channel_sizes, kernel_sizes, np_values, ch_kernel_collections)
plot_speedup_comparison(channel_sizes, kernel_sizes, np_values, ch_kernel_collections)
plot_speedup_simple_vs_chwise(channel_sizes, kernel_sizes, np_values, ch_kernel_collections)
plot_speedup_simple_vs_krnwise(channel_sizes, kernel_sizes, np_values, ch_kernel_collections)


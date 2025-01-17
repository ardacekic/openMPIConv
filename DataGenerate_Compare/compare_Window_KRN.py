import numpy as np
import re

def compare_outputs(file1, file2):
    """
    Compares two output files line by line.

    :param file1: Path to the first output file
    :param file2: Path to the second output file
    """
    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        if len(lines1) != len(lines2):
            print("Mismatch in number of lines.")
            return

        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            if line1.strip() != line2.strip():
                print(f"Difference found on line {i + 1}:")
                print(f"File 1: {line1.strip()}")
                print(f"File 2: {line2.strip()}")
                return

        print("Files match perfectly.")

# Example usage
compare_outputs("mpi_convolution_output.txt", "mpi_convolution_output_normal.txt")
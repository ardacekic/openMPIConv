
# OpenMPI Convolution Operations

## Overview 
This repository contains implementations of convolution operations using OpenMPI to demonstrate the efficiency of parallel computing. The code is organized into several C programs that perform channel-wise and kernel-wise convolutions, optimized with BLAS for enhanced performance.

## Features
Channel-wise Convolution: Distributes image channels across MPI nodes.  
Kernel-wise Convolution: Distributes convolution kernels across MPI nodes.  
Performance Comparison: Python scripts to compare execution times and efficiency.  

## Dependencies
MPI (Message Passing Interface)  
BLAS (Basic Linear Algebra Subprograms)  

## Getting Started  
To run the programs, clone this repository and use the provided shell scripts:  

    git clone https://github.com/ardacekic/openMPIConv  
    cd openMPIConv  
    sh run.sh 


## Contribution 
Contributions are welcome. Please fork the repository and submit pull requests with your enhancements.

 


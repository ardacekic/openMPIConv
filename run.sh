!rm -rf mpi_convolution_output.txt
## RUN CONV USING PYTORCH ##
!python3 data_generate.py 
!python3 run_conv.py 
## RUN SIMPLE CONV USING C ##
!gcc -std=c99 -o conv.o simple_conv.c -lm
!./conv.o
## COMPARE SIMPLE METHOD
!python3 compare.py
## RUN CHWISE CONV USING C ##
!mpicc -o ch_conv_orig.o ch_orig.c 
!mpirun --allow-run-as-root --oversubscribe -np 2 ./ch_conv_orig.o
!python3 compare_parallel.py
## RUN KNWISE CONV USING C ##
!mpicc -o kernel_conv.o kernel_wise.c 
!mpirun --allow-run-as-root --oversubscribe -np 2 ./kernel_conv.o
!python3 compare_parallel_kernel.py
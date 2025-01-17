!apt-get update
!apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
!sudo apt-get install mpich
!pip install cython
!apt-get install build-essential  # Installs gcc
!mpicc -showme 
!mpicc -o ch_conv_orig.o mpi_conv.c -lm && mpirun --allow-run-as-root -np 2 ./ch_conv_orig.o
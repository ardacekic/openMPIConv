echo "Window Convolution Pipe"
python3 pipe.py

echo "Blas Pipe without Kernel-Wise Optimization"
python3 pipe_blas.py

echo "Blas Pipe Optimized"
python3 pipe_blas_optimized.py
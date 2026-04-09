# --- Execution Parameters ---
export OMP_NUM_THREADS ?= 16 
MPI_PROCS=2

# --- MPI Run Flags ---
MPIRUN_FLAGS = -np $(MPI_PROCS) \
               --bind-to none

# --- Compiler Flags ---
# Flags for MPI+OpenMP code
# Uncomment and add extra flags if you need them
MPI_OMP_EXTRA_CFLAGS = -march=native -mtune=native -flto -fno-math-errno
#MPI_OMP_EXTRA_LIBS =

# Flags for CUDA code
# Uncomment and add extra flags if you need them
CUDA_EXTRA_CFLAGS = -O3
#CUDA_EXTRA_LIBS =
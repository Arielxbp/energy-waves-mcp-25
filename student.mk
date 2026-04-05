# --- Execution Parameters ---
export OMP_NUM_THREADS ?= 4
MPI_PROCS=4

# --- MPI Run Flags ---
MPIRUN_FLAGS = -np $(MPI_PROCS) \
               --bind-to none

# --- Compiler Flags ---
# Flags for MPI+OpenMP code
# Uncomment and add extra flags if you need them
MPI_OMP_EXTRA_CFLAGS = -march=native -mtune=native
#MPI_OMP_EXTRA_LIBS =

# Flags for CUDA code
# Uncomment and add extra flags if you need them
CUDA_EXTRA_CFLAGS = --ftz=true --fmad=true -Xcompiler "-march=native -O3 -fno-math-errno -fno-trapping-math -ffinite-math-only"
#CUDA_EXTRA_LIBS =
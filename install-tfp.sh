TFP_VERSION=0.24.0
TF_VERSION=2.16.1

pip install --upgrade pip

pip install tensorflow[and-cuda]==$TF_VERSION
pip install tensorflow-probability[tf]==$TFP_VERSION tensorflow[and-cuda]==$TF_VERSION

# The following is a workaround for a bug in tensorflow cuda installation
# https://github.com/tensorflow/tensorflow/issues/63362#issuecomment-2134680575
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo '# Store original LD_LIBRARY_PATH 
export ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" 

# Get the CUDNN directory 
CUDNN_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)")))

# Set LD_LIBRARY_PATH to include CUDNN directory
export LD_LIBRARY_PATH=$(find ${CUDNN_DIR}/*/lib/ -type d -printf "%p:")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Get the ptxas directory  
PTXAS_DIR=$(dirname $(dirname $(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)")))

# Set PATH to include the directory containing ptxas
export PATH=$(find ${PTXAS_DIR}/*/bin/ -type d -printf "%p:")${PATH:+:${PATH}}

'>> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo '# Restore original LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}"

# Unset environment variables
unset CUDNN_DIR
unset PTXAS_DIR' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh


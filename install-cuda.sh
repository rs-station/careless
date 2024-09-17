ENVNAME=careless
PY_VERSION=3.11

conda activate base

result=$(conda create -n $ENVNAME python=$PY_VERSION 3>&2 2>&1 1>&3)

echo $result
if [[ $result == *"CondaSystemExit"* ]]; then
    echo "User aborted anaconda env creation. Exiting... "
    return
fi

conda activate $ENVNAME

# Install TensorFlow Probability
source <(curl -s https://raw.githubusercontent.com/rs-station/careless/main/install-tfp.sh)

# Reactivate to update cuda paths
conda activate $ENVNAME

# Install careless
pip install --upgrade careless

# Run careless devices
careless devices

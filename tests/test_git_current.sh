NAME=careless
URL=github.com/hekstra-lab/careless
PYTHON_VERSION=3.8

eval "$(conda shell.bash hook)"

conda create -yn $NAME python=$PYTHON_VERSION
conda activate $NAME
conda env list
git clone ssh://git@$URL
cd $NAME
pip install -e .
python setup.py test



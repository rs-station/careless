TFP_VERSION=0.25.0
TF_VERSION=2.18.0

pip install --upgrade pip

pip install tensorflow[and-cuda]==$TF_VERSION tf_keras
pip install tensorflow-probability[tf]==$TFP_VERSION tensorflow[and-cuda]==$TF_VERSION tf_keras


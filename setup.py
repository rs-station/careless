from setuptools import setup, find_packages

setup(
    name='careless',
    version='0.0.1',
    author='Kevin M. Dalton',
    author_email='kmdalton@fas.harvard.edu',
    packages=find_packages(),
    description='Scaling and merging crystallographic data with TensorFlow and Variational Inference',
    install_requires=[
        "numpy",
        "reciprocalspaceship",
        "tqdm",
        "h5py",
        "tables",
        "cloudpickle == 1.3.0",
        "tensorflow",
        "tensorflow-probability",
    ],
    scripts = [
            'careless/careless',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
)

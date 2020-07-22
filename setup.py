from setuptools import setup, find_packages

setup(
    name='careless',
    version='0.0.1',
    author='Kevin M. Dalton',
    author_email='kmdalton@fas.harvard.edu',
    packages=find_packages(),
    description='Scaling and merging crystallographic data with TensorFlow and Variational Inference',
    install_requires=[
        "numpy>=1.10.0,<1.19",
        "reciprocalspaceship >= 0.8.6",
        "tqdm",
        "h5py",
        "tables",
        "cloudpickle==1.3.0",
        "tensorflow>=2.1.0",
        "tensorflow-probability >= 0.09",
    ],
    scripts = [
            'careless/careless',
            'careless/laue_careless',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
)

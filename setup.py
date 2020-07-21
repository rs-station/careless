from setuptools import setup, find_packages

setup(
    name='careless',
    version='0.0.1',
    author='Kevin M. Dalton',
    author_email='kmdalton@fas.harvard.edu',
    packages=find_packages(),
    description='Scaling and merging crystallographic data with TensorFlow and Variational Inference',
    install_requires=[
        "reciprocalspaceship >= 0.8.6",
        "tensorflow",
        "tensorflow-probability >= 0.10",
        "numpy < 1.19.0, >= 1.16.0",
        "tqdm",
        "h5py",
        "tables",
    ],
    scripts = [
            'careless/careless',
            'careless/laue_careless',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
)

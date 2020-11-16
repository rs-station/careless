from setuptools import setup, find_packages

setup(
    name='careless',
    version='0.0.2',
    author='Kevin M. Dalton',
    author_email='kmdalton@fas.harvard.edu',
    packages=find_packages(),
    description='Scaling and merging crystallographic data with TensorFlow and Variational Inference',
    install_requires=[
        "numpy<1.19.0,>=1.16.0",
        "scipy==1.4.1",
        "reciprocalspaceship<=0.8.9",
        "tqdm",
        "h5py",
        "tables",
        "tensorflow",
        "tensorflow-probability",
    ],
    scripts = [
            'careless/careless',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
)

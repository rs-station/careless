from setuptools import setup, find_packages

setup(
    name='careless',
    version='0.1.4',
    author='Kevin M. Dalton',
    author_email='kmdalton@fas.harvard.edu',
    packages=find_packages(),
    description='Scaling and merging crystallographic data with TensorFlow and Variational Inference',
    install_requires=[
        "reciprocalspaceship>=0.9.8",
        "tqdm",
        "tensorflow>=2.4",
        "tensorflow-probability",
    ],
    scripts = [
            'careless/careless',
            'scripts/ccplot',
            'scripts/ccanom_plot',
            'scripts/make_difference_map',
            'scripts/optimistic_uncertainties',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
)

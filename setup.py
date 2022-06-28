from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("careless/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version


__version__ = getVersionNumber()

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/Hekstra-Lab/careless/issues",
    "Source Code": "https://github.com/Hekstra-Lab/careless",
}


LONG_DESCRIPTION = """
``careless`` is a command line utility for merging 
data from x-ray crystallography experiments. It 
can be used to simultaneously scale and merge integrated
reflection intensities from conventional, laue, 
and free-electron laser experiments. It uses approximate 
Bayesian inference and deep learning to estimate
merged structure factor amplitudes under a Wilson prior.
"""

setup(
    name="careless",
    version=__version__,
    author="Kevin M. Dalton",
    author_email="kmdalton@fas.harvard.edu",
    license="MIT",
    include_package_data=True,
    packages=find_packages(),
    long_description=LONG_DESCRIPTION,
    description="Merging crystallography data without much physics.",
    project_urls=PROJECT_URLS,
    python_requires=">=3.7,<3.11",
    url="https://github.com/Hekstra-Lab/careless",
    install_requires=[
        "reciprocalspaceship>=0.9.16",
        "tqdm",
        "tensorflow>=2.7.0rc1",
        "tensorflow-probability",
        "matplotlib",
    ],
    scripts=[
        "scripts/ccplot",
        "scripts/ccanom_plot",
        "scripts/make_difference_map",
        "scripts/stream2mtz",
    ],
    entry_points={
        "console_scripts": [
            "careless=careless.careless:main",
        ]
    },
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov", "pytest-xdist"],
)

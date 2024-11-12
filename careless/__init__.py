# Version number for careless
def getVersionNumber():
    version = None
    try:
        from setuptools.version import metadata

        version = metadata.version("careless")
    except ImportError:
        from setuptools.version import pkg_resources

        version = pkg_resources.require("careless")[0].version

    return version

__version__ = getVersionNumber()

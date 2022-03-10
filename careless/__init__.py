# Version number for careless
def getVersionNumber():
    import pkg_resources

    version = pkg_resources.require("careless")[0].version
    return version


__version__ = getVersionNumber()

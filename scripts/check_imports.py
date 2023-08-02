import importlib
import pkgutil
import sys
import traceback


def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        importlib.import_module(full_name)
        if recursive and is_pkg:
            import_submodules(full_name)
    return results


if __name__ == "__main__":
    import pauliopt

    import_submodules(pauliopt)
    print(sys.modules.keys())
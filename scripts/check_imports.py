import importlib
import pkgutil
import traceback


def import_submodules(package):
    """Import all submodules of a module, recursively, including subpackages.

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    try:
        if isinstance(package, str):
            package = importlib.import_module(package)
        results = {}
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_name = package.__name__ + "." + name
            importlib.import_module(full_name)
            if is_pkg:
                import_submodules(full_name)
        return results
    except ModuleNotFoundError as e:
        # Extracting and formatting the traceback for ModuleNotFoundError
        tb_info = traceback.extract_tb(e.__traceback__)
        last_entry = tb_info[-1]  # The last call before the error
        filename, line_number, func, text = last_entry
        error_message = f"ModuleNotFoundError occurred in file: {filename} at line: {line_number}, statement: {text}"
        print(error_message)
        exit(1)
    except Exception as e:
        # Extracting and formatting the traceback for all other exceptions
        tb_info = traceback.extract_tb(e.__traceback__)
        last_entry = tb_info[-1]  # The last call before the error
        filename, line_number, func, text = last_entry
        error_message = f"An exception occurred in file: {filename} at line: {line_number}, statement: {text}"
        print(error_message)
        exit(1)


if __name__ == "__main__":
    import pauliopt

    import_submodules(pauliopt)
    print("All imports successful!")

""" setup.py created according to https://packaging.python.org/tutorials/packaging-projects """

import setuptools  # type:ignore

setuptools.setup(
    name="pauliopt",
    version="0.0.4.post2",
    maintainer="sg495",
    maintainer_email="sg495@users.noreply.github.com",
    description="A Python library to simplify quantum circuits of Pauli gadgets.",
    url="https://github.com/sg495/pauliopt",
    packages=setuptools.find_packages(exclude=["test", "tests"]),
    classifiers=[  # see https://pypi.org/classifiers/
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
        "Typing :: Typed",
    ],
    package_data={"": [],
                  "pauliopt": ["pauliopt/py.typed"],
                  },
    include_package_data=True,
    install_requires=[
        "networkx",
        "numpy"
    ],
)

""" setup.py created according to https://packaging.python.org/tutorials/packaging-projects """

import setuptools #type:ignore

setuptools.setup(
    name="pauliopt",
    version="0.0.0",
    author="y-richie-y, sg495",
    author_email="y-richie-y@users.noreply.github.com",
    description="A Python library to simplify quantum circuits of Pauli gadgets.",
    url="https://github.com/sg495/pauliopt",
    packages=setuptools.find_packages(exclude=["test"]),
    classifiers=[ # see https://pypi.org/classifiers/
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "Natural Language :: English",
        "Typing :: Typed",
    ],
    package_data={"": [],
                  "pauliopt": ["pauliopt/py.typed"],
                 },
    include_package_data=True
)

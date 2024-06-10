from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PyDAE",
    version="0.0.2",
    author="Jonas Breuling",
    author_email="jonas.breuling@inm.uni-stuttgart.de",
    description="Python implementation of differential algebraic equation (DAE) solvers that should be added to scipy one day.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonasBreuling/PyDAE",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["numpy", "scipy", "matplotlib"],
)

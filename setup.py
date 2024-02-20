from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PyDAE",
    version="0.0.1",
    author="Jonas Breuling",
    author_email="jonas.breuling@inm.uni-stuttgart.de",
    description="Python interface to Radau5 and DASSL high-performance stiff ODE / DAE solvers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TODO",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "scipy"],
)

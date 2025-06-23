from setuptools import setup, find_packages

setup(
    name="rled",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "PyQt6>=6.5.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
) 
from setuptools import setup, find_packages


setup(
    name="polycraft_nov_data",
    version="0.0.1",
    description="Polycraft novelty data",
    url="https://github.com/tufts-ai-robotics-group/polycraft-novelty-data",
    author="Patrick Feeney, Sarah Schneider",
    packages=find_packages(),
    python_requires=">=3.8, <4",
    install_requires=[
        "matplotlib",
        "numpy",
        "scikit-image",
        "pandas",
        "torch",
        "torchvision",
        # dev packages, not installing correctly when in extras_require
        "autopep8",
        "flake8",
        "pep8-naming",
        "pytest",
        "setuptools",
    ],
)

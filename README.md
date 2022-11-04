# polycraft-novelty-data

[![polycraft-novelty-data](https://github.com/tufts-ai-robotics-group/polycraft-novelty-data/actions/workflows/main.yml/badge.svg)](https://github.com/tufts-ai-robotics-group/polycraft-novelty-data/actions/workflows/main.yml)

Visual novelty dataset for the Polycraft domain.

## Installation
If you do not have Pipenv installed, run the following in Python 3.10:
```
pip install pipenv
```
The dependencies can be installed within a Pipenv with the following commands:
```
pipenv install --categories "packages torch_cpu"
```
PyTorch may require different versions depending on the machine it is running on. The default command is for non-CUDA machines while swapping `torch_cpu` for `torch_cu116` installs PyTorch for CUDA 11.6. If a non-default version of PyTorch is required then generate the appropriate Pip command on the [PyTorch website](https://pytorch.org/get-started/locally/) then run it within the Pipenv by prepending ```pipenv run``` to it.

#### Conda Installation

The following instructions are **not recommended** unless you are unable to install a Python version compatible with the Pipenv.

For this installation PyTorch will be installed in the Conda environment using the appropriate command according to the [PyTorch website](https://pytorch.org/get-started/locally/). For example:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Then the Pipenv must be configured for the Conda environment before install:
```
pipenv --python=$(conda run which python) --site-packages
pipenv install
```

## Testing

To run unit tests, run the following command:
```
pipenv run python -m pytest
```

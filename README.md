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
pipenv install
```
Pytorch may require different versions depending on the machine it is running on. If a non-default version of Pytorch is required then generate the appropriate Pip command on the [Pytorch website](https://pytorch.org/get-started/locally/) then run it within the Pipenv by prepending ```pipenv run``` to it.

#### Conda Installation

The following instructions are **not recommended** unless you are unable to install a Python version compatible with the Pipenv.

For this installation Pytorch will be installed in the Conda environment using the appropriate command according to the [Pytorch website](https://pytorch.org/get-started/locally/). For example:
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

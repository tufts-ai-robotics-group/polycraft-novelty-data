# polycraft-novelty-data

Visual novelty datasets for the Polycraft domain.

## Installation
If you do not have Pipenv installed, run the following:
```
pip install pipenv
```
The Pipenv dependencies can be installed within a Pipenv with the following commands:
```
pipenv install
```
Pytorch requires different versions depending on the machine it is running on. Therefore it is not included in the Pipenv by default. To install Pytorch, generate the appropriate Pip command on the [Pytorch website](https://pytorch.org/get-started/locally/) then run it within the Pipenv by prepending ```pipenv run``` to it. For example:
```
pipenv run pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

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

## Adding Data to Repository

1. Create a folder within either the ```normal_data``` or ```novel_data``` folder in the ```datasets``` folder of the [SAIL-ON Box](https://tufts.app.box.com/folder/112726258179). Ensure the folder has a name that clearly describes how it is different from other data.

2. Upload the data to the Box folder.

3. Create a folder in the ```normal_data``` or ```novel_data``` folder of this repository with the same name as the Box folder.

4. Copy ```data_readme_template.md``` to the new folder as ```README.md```. Modify ```README.md``` with a title, description, and reproducable steps to generate the data.

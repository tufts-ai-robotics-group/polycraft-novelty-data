# polycraft-novelty-data

Visual novelty datasets for the Polycraft domain.

## Downloading Data

Data should be downloaded from the ```datasets``` folder of the [SAIL-ON Box](https://tufts.app.box.com/folder/112726258179).

## Installation
If you do not have Pipenv installed, run the following:
```
pip install pipenv
```
All dependencies can be installed within a Pipenv with the following commands:
```
pipenv install
```

## Adding Data to Repository

1. Create a folder within either the ```normal_data``` or ```novel_data``` folder in the ```datasets``` folder of the [SAIL-ON Box](https://tufts.app.box.com/folder/112726258179). Ensure the folder has a name that clearly describes how it is different from other data.

2. Upload the data to the Box folder.

3. Create a folder in the ```normal_data``` or ```novel_data``` folder of this repository with the same name as the Box folder.

4. Copy ```data_readme_template.md``` to the new folder as ```README.md```. Modify ```README.md``` with a title, description, and reproducable steps to generate the data.

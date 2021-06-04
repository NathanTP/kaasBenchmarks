
# First Time Setup

## Getting Models
You can use tools/getModels.py to fetch the various models we want to use. They
should total a few hundred MB. These will be stored in models/.

## Getting Data
Data is more complicated than the models, we are still working on an elegant
solution. For now, you can use the tools/make\_fake\_imagenet.sh to create a
simple dataset that is sufficient for testing. You can manually get other
datasets by following the instructions sprinkled throughout
https://github.com/mlcommons/inference

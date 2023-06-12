# CDH1 Cancer Discovery

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

ADD LINK TO PAPER.

## Reproducing Inference

### Requirements
- Python 3.8 or later
- CPU

### Setup
After cloning the repository to your local machine, install the code in editable mode from the repositories root directory `cdh1-cancer-dis/` by executing `pip install -e .`

### Test the setup
To test that the installation works, script a single aggregator model. This is done by executing `python -m cdh1_cancer_dis.model`

### Run the demo prediction
- 1. Script the ensemble model: 
  - `python -m cdh1_cancer_dis.script_ensemble`
  - This will create the torchscripted ensemble model we will require to run our demo predictions.
- 2. Predict the CDH1 on the provided demo embeddings using the torchscripted ensemble model:
  - `python -m cdh1_cancer_dis.predict`
  - The prediction step should print our the ground truth values, the model's continuous and binarized prediction. 

___
[Paige.AI CDH1 Cancer Discovery](https://github.com/Paige-AI/cdh1-cancer-dis) (c) by Paige.AI

[Paige.AI CDH1 Cancer Discovery](https://github.com/Paige-AI/cdh1-cancer-dis) is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this work. If not, see <http://creativecommons.org/licenses/by-nc-nd/4.0/>.
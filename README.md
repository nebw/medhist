# Medical history predicts phenome-wide disease onset and enables the rapid response to emerging health threats

This repository contains the implementation of the deep learning model used
in "Medical history predicts phenome-wide disease onset and enables the rapid 
response to emerging health threats".

## Repository contents

### Original code repository

The subfolder `ehrgraphs` contains the repository used for the models used in 
the publication. Included is a `environment.yml` file for setting up the conda
environment used for all experiments, a `setup.py` allowing for installation of
the package via `pip install ehrgraphs` with required dependencies.

To reproduce the results you will need to have access to UK Biobank and use
the preprocessing code in this repository: 
<https://github.com/JakobSteinfeldt/22_medical_records>

### Standalone implementation (Demo)

We also provide a standalone implementation with synthetic data in that can be used
to train a model similar to the one used in the publication, but without much of
the complexity of the original implementation.

We recommend adapting this standalone implementation if you are trying to build a 
similar model for your data.

The standalone implementation can be found in `medical_history_model_standalone.ipynb`.

## System requirements

The models were trained on HPC nodes with 64 cpu cores, 512 GB RAM, and one 
NVIDIA A100 80G GPU. 

## Reference

tbd


# Feature encoding

This repository provides the code for the feature encoding analysis in ([Nonaka et al., 2021](https://doi.org/10.1016/j.isci.2021.103013)) to predict fMRI responses from DNN features.

## Hands-on tutorial

- [handson/01_encoding.ipynb](handson/01_encoding.ipynb) \[[Google Colab](https://colab.research.google.com/github/KamitaniLab/feature-encoding/blob/master/handson/01_encoding.ipynb)\]

## Usage

### Environment setup

See [requirements.txt](requirements.txt) for the required Python packages.

Here is example commands to setup environment using venv.

```shell
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

[Pycortex](https://github.com/gallantlab/pycortex) are optionally required for visualization of voxel-wise encoding accuracy in [evaluation.ipynb](evaluation.ipynb).

### 

```shell
# Training of encoding models
python train_encoding_fastl2lir.py <config.yaml>

# Prediction of fMRI responses
python predict_fmri_fastl2lir.py <config.yaml>

# Evaluation
jupyter notebook evaluation.ipynb
```

## Reproduction of Nonaka et al. (2021)

### Environment setup

Please follow the instruction in [Environment setup](#environment-setup).

### Data setup

```shell
# In "./data" directory:

# fMRI data (collected by Shen et al., 2019)
python download.py fmri_deeprecon_fmriprep_hcpvc 

# [Optional] Pycortex data for visualization
python download.py pycortex
```

### Encoding analysis

Command:

```shell
# Training of encoding models
python train_encoding_fastl2lir.py config/bhscore_encoding_fmriprep_hcprois.yaml -o +network=<network name>

# Prediction of fMRI responses
python predict_fmri_fastl2lir.py config/bhscore_encoding_fmriprep_hcprois.yaml -o +network=<network name>
```

Example:

```shell
# Training of encoding models
python train_encoding_fastl2lir.py config/bhscore_encoding_fmriprep_hcprois.yaml -o +network=AlexNet

# Prediction of fMRI responses
python predict_fmri_fastl2lir.py config/bhscore_encoding_fmriprep_hcprois.yaml -o +network=AlexNet
```

Avaniable networks:

- Alexnet
- VGG19_nonaka
- VGG16_nonaka

## References

- Nonaka, Majima, Aoki, and Kamitani (2021) Brain hierarchy score: Which deep neural networks are hierarchically brain-like? *iScience*. [10.1016/j.isci.2021.103013](https://doi.org/10.1016/j.isci.2021.103013)
- Shen, Horikawa, Majima, Kamitani (2019) Deep image reconstruction from human brain activity. *PLoS Comput. Biol*. [10.1371/journal.pcbi.1006633](https://doi.org/10.1371/journal.pcbi.1006633)

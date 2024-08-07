# Feature encoding

This repository provides the code for the feature encoding analysis to predict fMRI responses from DNN features.
The encoding analysis is used in the following study:

- Nonaka, Majima, Aoki, and Kamitani (2021) Brain hierarchy score: Which deep neural networks are hierarchically brain-like? *iScience*. [10.1016/j.isci.2021.103013](https://doi.org/10.1016/j.isci.2021.103013)

## Preparation

### Environment setup

See [requirements.txt](requirements.txt) for the required Python packages.

Here is example commands to setup environment using venv.

```shell
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

[Pycortex](https://github.com/gallantlab/pycortex) are optionally required for visualization of voxel-wise encoding accuracy in [evaluation.ipynb](evaluation.ipynb).

### Data setup

You can download the data used in the encoding analysis by the following command.

```shell
# In "./data" directory:
python download.py <data name>
```

The following data are available:

- fMRI data
  - `fmri_deeprecon_fmriprep_hcpvc`: fMRI data of the visual cortex (VC) used in the encoding analysis.
  - `fmri_deeprecon_fmriprep`: fMRI data of the whole brain used in the encoding analysis (not used in the example analysis).
- DNN features
  - `features_imagenet_training_vgg19_random5000`: VGG-19 features of ImageNet images used as training stimuli in the fMRI experiment; randomly selected 5000 units in each layer.
  - `features_imagenet_test_vgg19_random5000`: VGG-19 features of ImageNet images used as test stimuli in the fMRI experiment; randomly selected 5000 units in each layer.
  - `features_imagenet_training_vgg19`: VGG-19 features of ImageNet images used as training stimuli in the fMRI experiment; all units.
  - `features_imagenet_test_vgg19`: VGG-19 features of ImageNet images used as test stimuli in the fMRI experiment; all units.

Please download data required in the analysis you want to run (see "Encoding analysis").

Example commands:

```shell
# In "./data" directory:

# fMRI data (visual cortex only)
python download.py fmri_deeprecon_fmriprep_hcpvc

# VGG-19 features of ImageNet (random 5000 units)
python download.py features_imagenet_training_vgg19_random5000
python download.py features_imagenet_test_vgg19_random5000

# VGG-19 features of ImageNet (all units)
python download.py features_imagenet_training_vgg19
python download.py features_imagenet_test_vgg19
```

## Encoding analysis

### Using randomly selected 5000 units in each layer

This is light-weighted version of the encoding analysis, which uses randomly selected 5000 units in each layer of VGG-19 instead of all units. The analysis code is expected to run on a typical personal computer.

Requried data:

- `fmri_deeprecon_fmriprep_vc`
- `features_imagenet_training_vgg19_random5000`
- `features_imagenet_test_vgg19_random5000`

```shell
# Training of encoding models
python train_encoding_fastl2lir.py config/encoding_deeprecon_hcp_rois_vgg19_random5000_pyfastl2lir_alpha100_select500units.yaml

# Prediction of fMRI responses
python predict_fmri_fastl2lir.py  config/encoding_deeprecon_hcp_rois_vgg19_random5000_pyfastl2lir_alpha100_select500units.yaml
```

Evaluation: see [evaluation.ipynb](evaluation.ipynb)

### Using all DNN units

This version uses all units in each layer of VGG-19. Note that this requires a large amount of RAM (~100GB) and may only be runnable in HPCs.

Requried data:

- `fmri_deeprecon_fmriprep_vc`
- `features_imagenet_training_vgg19`
- `features_imagenet_test_vgg19`

```shell
# Training of encoding models
python train_encoding_fastl2lir.py config/encoding_deeprecon_hcp_rois_vgg19_pyfastl2lir_alpha100_select500units.yaml

# Prediction of fMRI responses
python predict_fmri_fastl2lir.py  config/encoding_deeprecon_hcp_rois_vgg19_pyfastl2lir_alpha100_select500units.yaml
```

Evaluation: see [evaluation.ipynb](evaluation.ipynb) (need modification)

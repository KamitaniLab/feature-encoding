# Feature encoding

This repository provides the code for the feature encoding analysis in ([Nonaka et al., 2021](https://doi.org/10.1016/j.isci.2021.103013)) to predict fMRI responses from DNN features.

- [Usage](#usage)
- [Hands-on tutorials](#hands-on-tutorials)
- [Reproduction of Nonaka et al. (2021)](#reproduction-of-nonaka-et-al-2021)

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

### Training of the encoding models and prediction of fMRI responses

```shell
# Training of encoding models
python train_encoding_fastl2lir.py <config.yaml>

# Prediction of fMRI responses
python predict_fmri_fastl2lir.py <config.yaml>

# Evaluation
jupyter notebook evaluation.ipynb
```

### Examples

```shell
# Download example data
# In ./data directory
python download.py fmri_deeprecon_fmriprep_hcpvc
python download.py features_imagenet_training_vgg19_random5000 
python download.py features_imagenet_test_vgg19_random5000 

# Traning and test
python train_encoding_fastl2lir.py config/example_encoding_deeprecon_hcp_rois_vgg19_random5000_pyfastl2lir_alpha100_select500units.yaml
python predict_fmri_fastl2lir.py config/example_encoding_deeprecon_hcp_rois_vgg19_random5000_pyfastl2lir_alpha100_select500units.yaml

# Evaluation (iPython notebook)
jupyter notebook evaluation.ipynb
```

## Hands-on tutorials

These iPython notebooks provides hands-on tutorials for the encoding analysis (DNN features to fMRI responses) and calculation of the brain hierarchically (BH) score based on the encoding and decoding results (Nonaka et al., 2021).

- [handson/01_encoding.ipynb](handson/01_encoding.ipynb) \[[Google Colab](https://colab.research.google.com/github/KamitaniLab/feature-encoding/blob/master/handson/01_encoding.ipynb)\]
- [handson/02_bhscore.ipynb](handson/02_bhscore.ipynb) \[[Google Colab](https://colab.research.google.com/github/KamitaniLab/feature-encoding/blob/master/handson/02_bhscore.ipynb)\]

## Reproduction of Nonaka et al. (2021)

Here we describe how to reproduce the encoding analysis in Nonaka et al. (2021) using the feature-encoding codebase.

Note: the encoding analysis with all units in each DNN layer is very memory-intensive and only executable on HPCs with large RAM.
Less memory-intensive version will be added soon.

### Environment setup

Please follow the instruction in [Environment setup](#environment-setup).

### Data setup

```shell
# In "./data" directory:

# fMRI data (collected by Shen et al., 2019)
python download.py fmri_deeprecon_fmriprep_hcpvc 
python download.py <DNN feature dataset>
```

Available DNN feature datasets (not all of them are available yet):

- features_bhscore_allunits_caffe-AlexNet
- [TBA] features_bhscore_allunits_caffe-DenseNet_121
- [TBA] features_bhscore_allunits_caffe-DenseNet_161
- [TBA] features_bhscore_allunits_caffe-DenseNet_169
- [TBA] features_bhscore_allunits_caffe-DenseNet_201
- [TBA] features_bhscore_allunits_caffe-InceptionResNet-v2
- [TBA] features_bhscore_allunits_caffe-SqueeseNet
- [TBA] features_bhscore_allunits_caffe-SqueeseNet1.0
- features_bhscore_allunits_caffe-VGG-F
- features_bhscore_allunits_caffe-VGG-M
- features_bhscore_allunits_caffe-VGG-S
- features_bhscore_allunits_caffe-VGG16_nonaka
- features_bhscore_allunits_caffe-VGG19_nonaka
- features_bhscore_allunits_pytorch-CORnet_R
- features_bhscore_allunits_pytorch-CORnet_S
- features_bhscore_allunits_pytorch-CORnet_Z
- features_bhscore_allunits_pytorch-resnet18
- features_bhscore_allunits_pytorch-resnet34
- features_bhscore_allunits_tensorflow-inception_v1
- features_bhscore_allunits_tensorflow-inception_v2
- features_bhscore_allunits_tensorflow-inception_v3
- features_bhscore_allunits_tensorflow-inception_v4
- features_bhscore_allunits_tensorflow-mobilenet_v2_1.4_224
- [TBA] features_bhscore_allunits_tensorflow-nasnet_large
- features_bhscore_allunits_tensorflow-nasnet_mobile
- [TBA] features_bhscore_allunits_tensorflow-pnasnet_large
- [TBA] features_bhscore_allunits_tensorflow-resnet_v2_101
- [TBA] features_bhscore_allunits_tensorflow-resnet_v2_152
- [TBA] features_bhscore_allunits_tensorflow-resnet_v2_50

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

- AlexNet
- DenseNet_121
- DenseNet_161
- DenseNet_169
- DenseNet_201
- InceptionResNet-v2
- SqueeseNet
- SqueeseNet1.0
- VGG-F
- VGG-M
- VGG-S
- VGG16_nonaka
- VGG19_nonaka
- CORnet_R
- CORnet_S
- CORnet_Z
- resnet18
- resnet34
- inception_v1
- inception_v2
- inception_v3
- inception_v4
- mobilenet_v2_1.4_224
- nasnet_large
- nasnet_mobile
- pnasnet_large
- resnet_v2_101
- resnet_v2_152
- resnet_v2_50

## References

- Nonaka, Majima, Aoki, and Kamitani (2021) Brain hierarchy score: Which deep neural networks are hierarchically brain-like? *iScience*. [10.1016/j.isci.2021.103013](https://doi.org/10.1016/j.isci.2021.103013)
- Shen, Horikawa, Majima, Kamitani (2019) Deep image reconstruction from human brain activity. *PLoS Comput. Biol*. [10.1371/journal.pcbi.1006633](https://doi.org/10.1371/journal.pcbi.1006633)

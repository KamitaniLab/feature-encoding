"""DNN Feature encoding - model training script."""


from typing import Dict, List, Optional

from itertools import product
import os
import shutil
from time import time
import warnings

import bdpy
from bdpy.bdata.utils import select_data_multi_bdatas, get_labels_multi_bdatas
from bdpy.dataform import Features, save_array
from bdpy.dataform.utils import get_multi_features
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTraining
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np
import yaml


# Main #######################################################################

def featenc_fastl2lir_train(
        fmri_data: Dict[str, List[str]],
        features_paths: List[str],
        output_dir: Optional[str] = './feature_encoders',
        rois: Optional[Dict[str, str]] = None,
        label_key: Optional[str] = None,
        layers: Optional[List[str]] = None,
        num_unit: Optional[Dict[str, int]] = None,
        feature_index_file: Optional[str] = None,
        alpha: int = 100,
        chunk_axis: int = 1,
        analysis_name: str = "feature_encoder_training"
):
    """Feature encoding model training.

    Parameters
    ----------
    fmri_data : dict
        Dictionary of fMRI data.
    features_paths : list of str
        List of paths to DNN features.
    output_dir : str, optional
        Output directory.
    rois : dict, optional
        Dictionary of ROIs.
    label_key : str, optional
        Key of the label in the fMRI data.
    layers : list of str, optional
        List of layer names.
    num_unit : dict, optional
        Dictionary of the number of selected units in each layer.
    feature_index_file : str, optional
        Path to the feature index file.
    alpha : int, optional
        Regularization parameter.
    chunk_axis : int, optional
        Chunk axis.
    analysis_name : str, optional
        Analysis name.

    Note
    ----
    If Y.ndim >= 3, Y is divided into chunks along `chunk_axis`.
    Note that Y[0] should be sample dimension.
    """
    if rois is None:
        rois = {}
    if layers is None:
        layers = []

    layers = layers[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Source features')
    print('Path:   %s' % features_paths)
    print('Layers: %s' % layers)
    print('Target brain data')
    print('Subjects:  %s' % list(fmri_data.keys()))
    print('ROIs:      %s' % list(rois.keys()))
    print('')

    # Load data --------------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    # FIXME: multiple datasets are not supported yet
    data_brain = {sbj: bdpy.BData(data_files[0]) for sbj, data_files in fmri_data.items()}

    if feature_index_file is not None:
        data_features = Features(features_paths[0], feature_index=os.path.join(features_paths[0], feature_index_file))
    else:
        data_features = Features(features_paths[0])

    # Initialize directories -------------------------------------------------
    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')

    # Save feature index -----------------------------------------------------
    if feature_index_file is not None:
        feature_index_save_file = os.path.join(output_dir, 'feature_index.mat')
        shutil.copy(feature_index_file, feature_index_save_file)
        print('Saved %s' % feature_index_save_file)

    # Analysis loop ----------------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for layer, sbj, roi in product(layers, fmri_data, rois):
        print('--------------------')
        print('Layer:     %s' % layer)
        print('Num units: %d' % num_unit[layer])
        print('Subject:   %s' % sbj)
        print('ROI:       %s' % roi)

        # Setup
        # -----
        analysis_id = analysis_name + '-' + sbj + '-' + roi + '-' + layer
        model_dir = os.path.join(output_dir, layer, sbj, roi, "model")
        makedir_ifnot(model_dir)

        # Check whether the analysis has been done or not.
        info_file = os.path.join(model_dir, 'info.yaml')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                info = yaml.safe_load(f)
            while info is None:
                warnings.warn('Failed to load info from %s. Retrying...'
                              % info_file)
                with open(info_file, 'r') as f:
                    info = yaml.safe_load(f)
            if '_status' in info and 'computation_status' in info['_status']:
                if info['_status']['computation_status'] == 'done':
                    print('%s is already done and skipped' % analysis_id)
                    continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # Brain data
        brain = data_brain[sbj].select(rois[roi])
        brain_labels = data_brain[sbj].get_label(label_key)

        # Features
        feat_labels = np.unique(brain_labels)
        feat = data_features.get(layer=layer, label=feat_labels)
        feat = feat.astype(np.float32)
        feat = feat.reshape(feat.shape[0], -1, order='F')

        # Use brain data that has a label included in feature data
        brain = np.vstack([_b for _b, bl in zip(brain, brain_labels) if bl in feat_labels])
        brain_labels = [bl for bl in brain_labels if bl in feat_labels]

        print('Elapsed time (data preparation): %f' % (time() - start_time))

        # Calculate normalization parameters
        # ----------------------------------

        # Normalize brain data
        brain_mean = np.mean(brain, axis=0)[np.newaxis, :]  # np.newaxis was added to match Matlab outputs
        brain_norm = np.std(brain, axis=0, ddof=1)[np.newaxis, :]

        # Normalize features
        feat_mean = np.mean(feat, axis=0)[np.newaxis, :]
        feat_norm = np.std(feat, axis=0, ddof=1)[np.newaxis, :]

        # Index to sort features by brain data (matching samples)
        # -----------------------------------------
        feat_index = np.array([np.where(np.array(feat_labels) == bl) for bl in brain_labels]).flatten()

        # Save normalization parameters
        # -----------------------------
        print('Saving normalization parameters.')
        norm_param = {
            'x_mean': feat_mean, 'y_mean': brain_mean,
            'x_norm': feat_norm, 'y_norm': brain_norm
        }
        save_targets = [u'x_mean', u'y_mean', u'x_norm', u'y_norm']
        for sv in save_targets:
            save_file = os.path.join(model_dir, sv + '.mat')
            if not os.path.exists(save_file):
                try:
                    save_array(save_file, norm_param[sv], key=sv, dtype=np.float32, sparse=False)
                    print('Saved %s' % save_file)
                except Exception:
                    warnings.warn('Failed to save %s. Possibly double running.' % save_file)

        # Preparing learning
        # ------------------
        model = FastL2LiR()
        model_param = {'alpha':  alpha,
                       'n_feat': num_unit[layer],
                       'dtype':  np.float32}

        # Distributed computation setup
        # -----------------------------
        makedir_ifnot('./tmp')
        distcomp_db = os.path.join('./tmp', analysis_name + '.db')
        distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

        # Model training
        # --------------
        print('Model training')
        start_time = time()

        train = ModelTraining(model, feat, brain)
        train.id = analysis_id
        train.model_parameters = model_param

        train.X_normalize = {'mean': feat_mean, 'std': feat_norm}
        train.Y_normalize = {'mean': brain_mean, 'std': brain_norm}
        train.X_sort = {'index': feat_index}

        train.dtype = np.float32
        train.chunk_axis = chunk_axis
        train.save_format = 'bdmodel'
        train.save_path = model_dir
        train.distcomp = distcomp

        train.run()

        print('Total elapsed time (model training): %f' % (time() - start_time))

    print('%s finished.' % analysis_name)

    return output_dir


# Entry point ################################################################

if __name__ == '__main__':

    cfg = init_hydra_cfg()

    analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]

    training_fmri = {
        subject["name"]: subject["paths"]
        for subject in cfg["encoder"]["fmri"]["subjects"]
    }
    rois = {
        roi["name"]: roi["select"]
        for roi in cfg["encoder"]["fmri"]["rois"]
    }
    label_key = cfg["encoder"]["fmri"]["label_key"]

    training_features = cfg["encoder"]["features"]["paths"]
    layers = [layer["name"] for layer in cfg["encoder"]["features"]["layers"]]
    num_unit = {
        layer["name"]: layer["num"]
        for layer in cfg["encoder"]["features"]["layers"]
    }
    feature_index_file = cfg.encoder.features.get("index_file", None)

    encoder_dir = cfg["encoder"]["path"]

    featenc_fastl2lir_train(
        training_fmri,
        training_features,
        output_dir=encoder_dir,
        rois=rois,
        label_key=label_key,
        layers=layers,
        num_unit=num_unit,
        feature_index_file=feature_index_file,
        alpha=cfg["encoder"]["parameters"]["alpha"],
        analysis_name=analysis_name
    )

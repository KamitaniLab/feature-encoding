"""DNN Feature encoding - fMRI prediction script"""


from typing import Dict, List, Optional

from itertools import product
import os
import shutil
from time import time

import bdpy
from bdpy.dataform import Features, load_array, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTest
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np


# Main #######################################################################

def featenc_fastl2lir_predict(
        features_paths,
        encoder_path,
        output_dir: Optional[str] = './encoded_fmri',
        subjects: Optional[str] = None,
        rois: Optional[str] = None,
        layers: Optional[str] = None,
        feature_index_file=None,
        chunk_axis: Optional[int] = 1,
        analysis_name: Optional[str] = "fmr_prediction"
):
    """Feature prediction.

    Parameters
    ----------
    features_paths : List[str]
        List of paths to the feature data.
    encoder_path : str
        Path to the encoder directory.
    output_dir : str, optional
        Output directory.
    subjects : List[str], optional
        List of subjects.
    rois : List[str], optional
        List of ROIs.
    layers : List[str], optional
        List of layers.
    feature_index_file : str, optional
        Feature index file.
    chunk_axis : int, optional
        Chunk axis.
    analysis_name : str, optional
        Analysis name.
    """
    if subjects is None:
        subjects = []
    if rois is None:
        rois = []
    if layers is None:
        layers = []
    
    layers = layers[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Encoders:  %s' % encoder_path)
    print('Subjects:  %s' % subjects)
    print('ROIs:      %s' % rois)
    print('Source features')
    print('Path:   %s' % features_paths)
    print('Layers: %s' % layers)
    print('')

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    # FIXME: multiple datasets are not supported yet
    if feature_index_file is not None:
        data_features = Features(features_paths[0], feature_index=os.path.join(features_paths[0], feature_index_file))
    else:
        data_features = Features(features_paths[0])

    # Initialize directories -------------------------------------------
    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')

    # Save feature index -----------------------------------------------------
    if feature_index_file is not None:
        feature_index_save_file = os.path.join(output_dir, 'feature_index.mat')
        shutil.copy(feature_index_file, feature_index_save_file)
        print('Saved %s' % feature_index_save_file)

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for layer, sbj, roi in product(layers, subjects, rois):
        print('--------------------')
        print('Layer:   %s' % layer)
        print('Subject: %s' % sbj)
        print('ROI:     %s' % roi)

        # Distributed computation setup
        # -----------------------------
        analysis_id = analysis_name + '-' + sbj + '-' + roi + '-' + layer
        results_dir_prediction = os.path.join(output_dir, layer, sbj, roi)

        if os.path.exists(results_dir_prediction):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        makedir_ifnot(results_dir_prediction)

        distcomp_db = os.path.join('./tmp', analysis_name + '.db')
        distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)
        if not distcomp.lock(analysis_id):
            print('%s is already running. Skipped.' % analysis_id)
            continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # Features
        feat = data_features.get(layer=layer)
        feat = feat.astype(np.float32)
        feat = feat.reshape(feat.shape[0], -1, order='F')
        feat_labels = data_features.labels

        print('Elapsed time (data preparation): %f' % (time() - start_time))

        # Model directory
        # ---------------
        model_dir = os.path.join(encoder_path, layer, sbj, roi, "model")

        # Preprocessing
        # -------------
        feat_mean = load_array(os.path.join(model_dir, 'x_mean.mat'), key='x_mean')  # shape = (1, n_voxels)
        feat_norm = load_array(os.path.join(model_dir, 'x_norm.mat'), key='x_norm')  # shape = (1, n_voxels)
        brain_mean = load_array(os.path.join(model_dir, 'y_mean.mat'), key='y_mean')  # shape = (1, shape_features)
        brain_norm = load_array(os.path.join(model_dir, 'y_norm.mat'), key='y_norm')  # shape = (1, shape_features)

        feat = (feat - feat_mean) / feat_norm

        # Prediction
        # ----------
        print('Prediction')

        start_time = time()

        model = FastL2LiR()

        test = ModelTest(model, feat)
        test.model_format = 'bdmodel'
        test.model_path = model_dir
        test.dtype = np.float32
        test.chunk_axis = chunk_axis

        brain_pred = test.run()

        print('Total elapsed time (prediction): %f' % (time() - start_time))

        # Postprocessing
        # --------------
        brain_pred = brain_pred * brain_norm + brain_mean

        # Save results
        # ------------
        print('Saving results')

        start_time = time()

        # Predicted features
        for i, label in enumerate(feat_labels):
            # Predicted fMRI signal
            _brain = np.array([brain_pred[i,]])  # To make feat shape 1 x M x N x ...

            # Save file name
            save_file = os.path.join(results_dir_prediction, '%s.mat' % label)

            # Save
            save_array(save_file, _brain, key='fmri', dtype=np.float32, sparse=False)

        print('Saved %s' % results_dir_prediction)

        print('Elapsed time (saving results): %f' % (time() - start_time))

        distcomp.unlock(analysis_id)

    print('%s finished.' % analysis_name)

    return output_dir


# Entry point ################################################################

if __name__ == '__main__':

    cfg = init_hydra_cfg()

    analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]

    encoder_path = cfg["encoded_fmri"]["encoder"]["path"]
    subjects = [subject["name"] for subject in cfg["encoded_fmri"]["fmri"]["subjects"]]
    rois = [roi["name"] for roi in cfg["encoded_fmri"]["fmri"]["rois"]]

    test_features = cfg["encoded_fmri"]["features"]["paths"]
    layers = [layer["name"] for layer in cfg["encoded_fmri"]["features"]["layers"]]
    feature_index_file = cfg.encoded_fmri.features.get("index_file", None)
    
    featenc_fastl2lir_predict(
        test_features,
        encoder_path,
        output_dir=cfg["encoded_fmri"]["path"],
        subjects=subjects,
        rois=rois,
        layers=layers,
        feature_index_file=feature_index_file,
        analysis_name=analysis_name
    )

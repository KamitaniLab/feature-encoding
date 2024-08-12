'''DNN Feature decoding (corss-validation) - feature prediction script.'''


from typing import Dict, List, Optional

from itertools import product
import os
import shutil
from time import time

import bdpy
from bdpy.dataform import Features, load_array, save_array
from bdpy.dataform.utils import get_multi_features
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTest
from bdpy.ml.crossvalidation import make_cvindex_generator
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np


# Main #######################################################################

def featenc_cv_fastl2lir_predict(
        features_paths: List[str],
        fmri_data: Dict[str, List[str]],
        encoder_path: str,
        output_dir: str = './feature_decoding_cv',
        subjects: List[str] = [],
        rois: List[str] = [],
        label_key: Optional[str] = None,
        layers: Optional[List[str]] = None,
        feature_index_file: Optional[str] = None,
        cv_key: str ='Run',
        cv_folds: Optional[Dict[str, List[int]]] = None,
        cv_exclusive: Optional[str] = None,
        excluded_labels: List[str] = [],
        analysis_name: str = "feature_prediction"
):
    '''Cross-validation feature decoding.

    Input:

    - fmri_data
    - decoder_path

    Output:

    - output_dir

    Parameters:

    TBA

    '''
    if layers is None:
        layers = []

    layers = layers[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Source features')
    print('Path:     %s' % features_paths)
    print('Layers:   %s' % layers)
    print('Target brain data')
    print('Subjects: %s' % subjects)
    print('ROIs:     %s' % rois)
    print('CV:       %s' % cv_key)
    print('')

    # Load data --------------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: [bdpy.BData(f) for f in data_files] for sbj, data_files in fmri_data.items()}

    if feature_index_file is not None:
        data_features = [Features(f, feature_index=os.path.join(f, feature_index_file)) for f in features_paths]
    else:
        data_features = [Features(f) for f in features_paths]

    # Initialize directories -------------------------------------------------
    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')

    # Save feature index -----------------------------------------------------
    if feature_index_file is not None:
        feature_index_save_file = os.path.join(output_dir, 'feature_index.mat')
        shutil.copy(feature_index_file, feature_index_save_file)
        print('Saved %s' % feature_index_save_file)

    # Distributed computation setup ------------------------------------------
    distcomp_db = os.path.join('./tmp', analysis_name + '.db')
    distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

    # Analysis loop ----------------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for layer, sbj, roi in product(layers, subjects, rois):
        print('--------------------')
        print('Layer:   %s' % layer)
        print('Subject: %s' % sbj)
        print('ROI:     %s' % roi)

        # Cross-validation setup
        if cv_exclusive is not None:
            cv_exclusive_array = data_brain[sbj][0].select(cv_exclusive)  # FIXME: support multiple datasets
        else:
            cv_exclusive_array = None

        cv_index = make_cvindex_generator(
            data_brain[sbj][0].select(cv_key),  # FIXME: support multiple datasets
            folds=cv_folds,
            exclusive=cv_exclusive_array
        )
        if 'name' in cv_folds[0]:
            cv_labels = ['cv-{}'.format(cv['name']) for cv in cv_folds]
        else:
            cv_labels = ['cv-fold{}'.format(icv + 1) for icv in range(len(cv_folds))]

        for cv_label, (train_index, test_index) in zip(cv_labels, cv_index):
            print('CV fold: {} ({} training; {} test)'.format(cv_label, len(train_index), len(test_index)))

            # Setup
            # -----
            analysis_id = analysis_name + '-' + sbj + '-' + roi + '-' + cv_label + '-' + layer
            encoded_fmri_dir = os.path.join(output_dir, layer, sbj, roi, cv_label, 'encoded_fmri')

            if os.path.exists(encoded_fmri_dir):
                print('%s is already done. Skipped.' % analysis_id)
                continue

            makedir_ifnot(encoded_fmri_dir)

            if not distcomp.lock(analysis_id):
                print('%s is already running. Skipped.' % analysis_id)

            # Preparing data
            # --------------
            print('Preparing data')

            start_time = time()

            # Features
            brain_labels = data_brain[sbj][0].get_label(label_key)  # FIXME: support multiple datasets
            brain_labels = np.array(brain_labels)[test_index]
            brain_labels = [label for label in brain_labels if label not in excluded_labels]
            feat_labels = np.unique(brain_labels)
            feat = get_multi_features(data_features, layer, labels=feat_labels)

            print('Elapsed time (data preparation): %f' % (time() - start_time))

            # Model directory
            # ---------------
            model_dir = os.path.join(encoder_path, layer, sbj, roi, cv_label, 'model')

            # Preprocessing
            # -------------
            feat_mean = load_array(os.path.join(model_dir, 'x_mean.mat'), key='x_mean')
            feat_norm = load_array(os.path.join(model_dir, 'x_norm.mat'), key='x_norm')
            brain_mean = load_array(os.path.join(model_dir, 'y_mean.mat'), key='y_mean')
            brain_norm = load_array(os.path.join(model_dir, 'y_norm.mat'), key='y_norm')

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
                # Predicted features
                _brain = np.array([brain_pred[i,]])  # To make feat shape 1 x M x N x ...

                # Save file name
                save_file = os.path.join(encoded_fmri_dir, '%s.mat' % label)

                # Save
                save_array(save_file, _brain, key='feat', dtype=np.float32, sparse=False)

            print('Saved %s' % encoded_fmri_dir)

            print('Elapsed time (saving results): %f' % (time() - start_time))

            distcomp.unlock(analysis_id)

    print('%s finished.' % analysis_name)

    return output_dir


# Entry point ################################################################

if __name__ == '__main__':

    cfg = init_hydra_cfg()

    analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]

    encoder_path = cfg["encoded_fmri"]["encoder"]["path"]

    test_fmri_data = {
        subject["name"]: subject["paths"]
        for subject in cfg["encoded_fmri"]["fmri"]["subjects"]
    }
    rois = {
        roi["name"]: roi["select"]
        for roi in cfg["encoded_fmri"]["fmri"]["rois"]
    }
    label_key = cfg["encoded_fmri"]["fmri"]["label_key"]

    test_feautres = cfg["encoded_fmri"]["features"]["paths"]
    layers = [layer["name"] for layer in cfg["encoder"]["features"]["layers"]]
    feature_index_file = cfg.encoder.features.get("index_file", None)

    encoded_fmri_dir = cfg["encoded_fmri"]["path"]

    excluded_labels = cfg.encoded_fmri.fmri.get("exclude_labels", [])

    cv_folds = cfg.cv.get("folds", None)
    cv_exclusive = cfg.cv.get("exclusive_key", None)

    featenc_cv_fastl2lir_predict(
        test_feautres,
        test_fmri_data,
        encoder_path,
        output_dir=encoded_fmri_dir,
        subjects=list(test_fmri_data.keys()),
        rois=rois,
        label_key=label_key,
        layers=layers,
        feature_index_file=feature_index_file,
        excluded_labels=excluded_labels,
        cv_key=cfg["cv"]["key"],
        cv_folds=cv_folds,
        cv_exclusive=cv_exclusive,
        analysis_name=analysis_name
    )

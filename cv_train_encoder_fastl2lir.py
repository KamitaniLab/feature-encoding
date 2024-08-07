'''Feature encoding (corss-validation) - encoders training script.'''


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
from bdpy.ml.crossvalidation import make_cvindex_generator
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np
import yaml


# Main #######################################################################

def featenc_cv_fastl2lir_train(
        fmri_data: Dict[str, List[str]],
        features_paths: List[str],
        output_dir: str = './feature_encoding_cv',
        rois: Optional[Dict[str, str]] = None,
        label_key: Optional[str] = None,
        layers: Optional[List[str]] = None,
        num_unit: Optional[Dict[str, int]] = None,
        feature_index_file: Optional[str] = None,
        cv_key: str = 'Run',
        cv_folds: Optional[Dict[str, List[int]]] = None,
        cv_exclusive: Optional[str] = None,
        alpha: int = 100,
        analysis_name: str = "feature_encoder_training"
):
    '''Cross-validation feature encoder training.

    Input:

    - fmri_data
    - features_paths

    Output:

    - output_dir

    Parameters:

    TBA

    '''
    layers = layers[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Source features')
    print('Path:   %s' % features_paths)
    print('Layers: %s' % layers)
    print('Target brain data')
    print('Subjects:  %s' % list(fmri_data.keys()))
    print('ROIs:      %s' % list(rois.keys()))
    print('CV:              %s' % cv_key)
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
    makedir_ifnot('./tmp')
    distcomp_db = os.path.join('./tmp', analysis_name + '.db')
    distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

    # Analysis loop ----------------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for layer, sbj, roi in product(layers, fmri_data, rois):
        print('--------------------')
        print('Layer:     %s' % layer)
        print('Num units: %d' % num_unit[layer])
        print('Subject:   %s' % sbj)
        print('ROI:       %s' % roi)

        # Cross-validation setup
        if cv_exclusive is not None:
            # FIXME: support multiple datasets
            cv_exclusive_array = data_brain[sbj][0].select(cv_exclusive)
        else:
            cv_exclusive_array = None

        # FXIME: support multiple datasets
        cv_index = make_cvindex_generator(
            data_brain[sbj][0].select(cv_key),
            folds=cv_folds,
            exclusive=cv_exclusive_array
        )

        for icv, (train_index, test_index) in enumerate(cv_index):
            print('CV fold: {} ({} training; {} test)'.format(icv + 1, len(train_index), len(test_index)))

            # Setup
            # -----
            analysis_id = analysis_name + '-' + sbj + '-' + roi + '-' + str(icv + 1) + '-' + layer
            model_dir   = os.path.join(output_dir, layer, sbj, roi, 'cv-fold{}'.format(icv + 1), 'model')

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
            brain = select_data_multi_bdatas(data_brain[sbj], rois[roi])
            brain_labels = get_labels_multi_bdatas(data_brain[sbj], label_key)

            brain = brain[train_index, :]
            brain_labels = np.array(brain_labels)[train_index]

            # Features
            feat_labels = np.unique(brain_labels)
            feat = get_multi_features(data_features, layer, labels=feat_labels)

            # Use brain data that has a label included in feature data
            brain = np.vstack([_b for _b, bl in zip(brain, brain_labels) if bl in feat_labels])
            brain_labels = [bl for bl in brain_labels if bl in feat_labels]

            # Index to sort features by brain data (matching samples)
            feat_index = np.array([np.where(np.array(feat_labels) == bl) for bl in brain_labels]).flatten()

            # Get training samples of Y
            feat_train = feat[feat_index, :]

            print('Elapsed time (data preparation): %f' % (time() - start_time))

            # Calculate normalization parameters
            # ----------------------------------

            # Normalize X (fMRI data)
            brain_mean = np.mean(brain, axis=0)[np.newaxis, :]  # np.newaxis was added to match Matlab outputs
            brain_norm = np.std(brain, axis=0, ddof=1)[np.newaxis, :]

            # Normalize Y (DNN features)
            feat_mean = np.mean(feat_train, axis=0)[np.newaxis, :]
            feat_norm = np.std(feat_train, axis=0, ddof=1)[np.newaxis, :]

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
            model_param = {
                'alpha':  alpha,
                'n_feat': num_unit[layer],
                'dtype':  np.float32
            }

            # Model training
            # --------------
            print('Model training')
            start_time = time()

            train = ModelTraining(model, feat, brain)
            train.id = analysis_id
            train.model_parameters = model_param

            train.X_normalize = {'mean': feat_mean,  'std': feat_norm}
            train.Y_normalize = {'mean': brain_mean, 'std': brain_norm}
            train.X_sort = {'index': feat_index}

            train.dtype = np.float32
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

    cv_folds = cfg.cv.get("folds", None)
    cv_exclusive = cfg.cv.get("exclusive_key", None)

    featenc_cv_fastl2lir_train(
        training_fmri,
        training_features,
        output_dir=encoder_dir,
        rois=rois,
        label_key=label_key,
        layers=layers,
        num_unit=num_unit,
        feature_index_file=feature_index_file,
        alpha=cfg["encoder"]["parameters"]["alpha"],
        cv_key=cfg["cv"]["key"],
        cv_folds=cv_folds,
        cv_exclusive=cv_exclusive,
        analysis_name=analysis_name
    )

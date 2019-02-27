
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import os.path
import numpy as np
import struct


def normalize_feature(embedding):
    for i in range(embedding.shape[0]):
        _norm = np.linalg.norm(embedding[i])
        embedding[i] /= _norm
        # embedding[i] = embedding[i]/np.linalg.norm(embedding[i])
    return embedding


def write_bin(path, feature):
    save_dir, _ = os.path.split(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))

    return


def save_features(model, dataloader_ori, dataloader_flip,
                  save_dir, dataset, feature_ext,
                  fs_noise_map=None, mf_noise_map=None, save_original=False):
    start = datetime.now()

    if dataset == 'megaface':
        dataset_out_dir = os.path.join(save_dir, 'MegaFace_Features')
    elif dataset == 'facescrub':
        dataset_out_dir = os.path.join(save_dir, 'FaceScrub_Features')
    else:
        raise ValueError

    if fs_noise_map is not None:
        fname2center = {}
        noises = []

    succ = 0
    niter = 0

    for i, ((images_ori, batch_image_path), (images_flip, batch_image_path)) in enumerate(zip(dataloader_ori, dataloader_flip)):

        features_ori = model(images_ori).cpu().numpy()
        features_flip = model(images_flip).cpu().numpy()
        features = normalize_feature(features_ori+features_flip)

        for n in range(features.shape[0]):
            feature = features[n].copy()
            feature_dim = feature.shape[0]
            image_path = batch_image_path[n]
            _path = image_path.split('/')
            if dataset == 'facescrub':
                a, b = _path[-2], _path[-1]
                out_path = os.path.join(dataset_out_dir, a, b+".bin")
                if save_original:
                    write_bin(out_path, feature)
                out_path_remove_noise = out_path.replace('_Features', '_Features_cm')
                if b not in fs_noise_map:
                    feature_cm = np.full((feature_dim+feature_ext,), 0.0, dtype=np.float32)
                    feature_cm[0: feature_dim] = feature.copy()
                    write_bin(out_path_remove_noise, feature_cm)
                    if a not in fname2center:
                        fname2center[a] = np.zeros((feature_dim+feature_ext,), dtype=np.float32)
                    fname2center[a] += feature_cm
                else:
                    noises.append((a, b))

            elif dataset == 'megaface':
                a1, a2, b = _path[-3], _path[-2], _path[-1]
                out_path = os.path.join(dataset_out_dir, a1, a2, b+".bin")
                if save_original:
                    write_bin(out_path, feature)
                # if mf_noise_map is not None:
                out_path_remove_noise = out_path.replace('_Features', '_Features_cm')
                bb = '/'.join([a1, a2, b])
                if bb not in mf_noise_map:
                    feature_cm = np.full((feature_dim+feature_ext,), 0.0, dtype=np.float32)
                    feature_cm[0: feature_dim] = feature.copy()
                    write_bin(out_path_remove_noise, feature_cm)
                else:
                    feature_cm = np.full((feature_dim+feature_ext,), 100.0, dtype=np.float32)
                    feature_cm[0: feature_dim] = feature.copy()
                    write_bin(out_path_remove_noise, feature_cm)
                if not os.path.exists(out_path_remove_noise):
                    raise ValueError
            succ += 1
            if succ % 100000 == 0:
                print("succ", niter, succ, out_path)
        niter += 1
        if niter % 100 == 0:
            print("niter", niter, succ)

    if fs_noise_map is not None:
        print('noises', len(noises))
        for k in noises:
            a, b = k
            out_dir = os.path.join(dataset_out_dir, a)
            assert a in fname2center
            center = fname2center[a]
            g = np.zeros((feature_dim+feature_ext,), dtype=np.float32)
            g2 = np.random.uniform(-0.001, 0.001, (feature_dim,))
            g[0:feature_dim] = g2
            f = center+g
            _norm = np.linalg.norm(f)
            f /= _norm
            out_path_remove_noise = os.path.join(out_dir.replace('_Features', '_Features_cm'),
                                                 b+".bin")
            write_bin(out_path_remove_noise, f)
            succ += 1
        print("succ", niter, succ, out_path_remove_noise)

    print('total time', datetime.now()-start)
    print('features are saved in {}'.format(out_dir))

    return

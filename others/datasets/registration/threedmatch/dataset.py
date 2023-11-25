import re
import os.path as osp
import os
import pickle
import random
from typing import Dict
import numpy as np
import torch
import torch.utils.data
import cv2
from others.utils.img_utils import read_txt, read_intrinsic

from others.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)
from others.utils.registration import get_correspondences


class ThreeDMatchPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=0.0375,
        rotated=False,
    ):
        super(ThreeDMatchPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, 'metadata')
        self.data_root = osp.join(self.dataset_root, 'data')
        self.img_width = 160
        self.img_height = 120
        self.mean = np.asarray([0.406, 0.456, 0.485], dtype=np.float32)
        self.std = np.asarray([0.225, 0.224, 0.229], dtype=np.float32)
        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        points = torch.load(osp.join(self.data_root, file_name))
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def _load_img(self, rgb_file, depth_file):
        img_path = osp.join(self.data_root, rgb_file)
        depth_path = osp.join(self.data_root, depth_file)
        if osp.exists(img_path):
            img = cv2.imread(img_path)
        return img

    def _read_intrinsic(self, filepath):
        m = np.loadtxt(filepath, dtype=np.float32
        return m

    def _read_txt(self, filepath):
        sents = []
        extrinsic = []
        out = []
        filepath = osp.join(self.data_root, filepath)
        with open(filepath, 'r') as f:
            f = open(filepath, 'r')
            lines = f.readlines()
        for index in range(len(lines)):
            if index > 0:
                sents.append(lines[index])
        for line in sents:
            res = re.split('\t|\n', line)
            extrinsic.append(res)
        for ex in extrinsic:
            for index in range(len(ex)):
                if ex[index] == '':
                    continue
                ex[index] = float(ex[index])
                out.append(ex[index])
        s = np.asarray(out).reshape(4, 4)
        return s

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation, intrinsic):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        ref_intrinsic = intrinsic
        src_intrinsic = intrinsic
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
            ref_intrinsic = np.matmul(intrinsic, aug_rotation.T)

        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
            src_intrinsic = np.matmul(intrinsic, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation, ref_intrinsic, src_intrinsic

    def _resize_img(self, img, width, height):
        x_scale = img.shape[1] / width
        y_scale = img.shape[0] / height
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        return img, x_scale, y_scale

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        rotation = metadata['rotation']
        translation = metadata['translation']

        # get point cloud
        ref_points = self._load_point_cloud(metadata['pcd0'])
        src_points = self._load_point_cloud(metadata['pcd1'])
        ref_path = metadata['pcd0']
        src_path = metadata['pcd1']
        # get img
        ref_img_path = metadata['pcd0'][:-4] + "_color.png"
        src_img_path = metadata['pcd1'][:-4] + "_color.png"
        ref_depth_path = metadata['pcd0'][:-4] + "_depth.png"
        src_depth_path = metadata['pcd1'][:-4] + "_depth.png"
        ref_img = self._load_img(ref_img_path, ref_depth_path)
        src_img = self._load_img(src_img_path, src_depth_path)

        # resize image
        ref_img, ref_x_scale, ref_y_scale = self._resize_img(ref_img, self.img_width, self.img_height)
        src_img, src_x_scale, src_y_scale = self._resize_img(src_img, self.img_width, self.img_height)
        # normalize image
        ref_img = ref_img / 255.
        src_img = src_img / 255.
        ref_img = (ref_img - self.mean) / self.std
        src_img = (src_img - self.mean) / self.std
        # project points into image
        intrinsic_path = osp.abspath(osp.join(self.data_root, ref_path, '..', 'camera-intrinsics.txt'))
        ref_extrinsic_path = ref_path[:-4] + ".info.txt"
        src_extrinsic_path = src_path[:-4] + ".info.txt"
        intrinsic = self._read_intrinsic(intrinsic_path)
        ref_extrinsic = self._read_txt(ref_extrinsic_path)
        src_extrinsic = self._read_txt(src_extrinsic_path)
        ref_intrinsic = intrinsic
        src_intrinsic = intrinsic

        # augmentation
        if self.use_augmentation:
            ref_points, src_points, rotation, translation, ref_intrinsic, src_intrinsic = self._augment_point_cloud(
                    ref_points, src_points, rotation, translation, intrinsic)

        if self.rotated:
            ref_rotation = random_sample_rotation_v2()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_v2()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)

        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_img'] = ref_img.astype(np.float32)
        data_dict['src_img'] = src_img.astype(np.float32)
        data_dict['ref_extrinsic'] = ref_extrinsic.astype(np.float32)
        data_dict['src_extrinsic'] = src_extrinsic.astype(np.float32)
        data_dict['ref_intrinsic'] = ref_intrinsic.astype(np.float32)
        data_dict['src_intrinsic'] = src_intrinsic.astype(np.float32)
        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['ref_x_scale'] = ref_x_scale
        data_dict['ref_y_scale'] = ref_y_scale
        data_dict['src_x_scale'] = src_x_scale
        data_dict['src_y_scale'] = src_y_scale

        return data_dict

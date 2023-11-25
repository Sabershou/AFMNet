import os.path as osp
import random
import cv2
import numpy as np
import torch.utils.data

from others.utils.common import load_pickle
from others.utils.pointcloud import (
    random_sample_rotation,
    get_transform_from_rotation_translation,
    get_rotation_translation_from_transform,
)
from others.utils.registration import get_correspondences


class OdometryKittiPairDataset(torch.utils.data.Dataset):
    ODOMETRY_KITTI_DATA_SPLIT = {
        'train': ['00', '01', '02', '03', '04', '05'],
        'val': ['06', '07'],
        'test': ['08', '09', '10'],
    }

    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_min_scale=0.8,
        augmentation_max_scale=1.2,
        augmentation_shift=2.0,
        augmentation_rotation=1.0,
        return_corr_indices=False,
        matching_radius=None,
    ):
        super(OdometryKittiPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit
        self.img_width = 320
        self.img_height = 160
        self.mean = np.asarray([0.406, 0.456, 0.485], dtype=np.float32)
        self.std = np.asarray([0.225, 0.224, 0.229], dtype=np.float32)

        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_min_scale = augmentation_min_scale
        self.augmentation_max_scale = augmentation_max_scale
        self.augmentation_shift = augmentation_shift
        self.augmentation_rotation = augmentation_rotation

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.metadata = load_pickle(osp.join(self.dataset_root, 'metadata', f'{subset}.pkl'))

    def _augment_point_cloud(self, ref_points, src_points, transform, proj_matrix):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        ref_points = ref_points + (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.augmentation_noise
        src_points = src_points + (np.random.rand(src_points.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        ref_proj_matrix = proj_matrix
        src_proj_matrix = proj_matrix
        aug_rotation = random_sample_rotation(self.augmentation_rotation)
        aug_rotation_pro = np.concatenate([aug_rotation, np.zeros([3, 1])], axis=1)
        a = np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 4)
        aug_rotation_pro = np.concatenate([aug_rotation_pro, a], axis=0)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
            ref_proj_matrix = np.matmul(proj_matrix, aug_rotation_pro.T)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
            src_proj_matrix = np.matmul(proj_matrix, aug_rotation_pro.T)
        # random scaling
        scale = random.random()
        scale = self.augmentation_min_scale + (self.augmentation_max_scale - self.augmentation_min_scale) * scale
        ref_points = ref_points * scale
        src_points = src_points * scale
        translation = translation * scale
        # # random shift
        # ref_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        # src_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        # ref_points = ref_points + ref_shift
        # src_points = src_points + src_shift
        # translation = -np.matmul(src_shift[None, :], rotation.T) + translation + ref_shift
        # compose transform from rotation and translation
        transform = get_transform_from_rotation_translation(rotation, translation)
        return ref_points, src_points, transform, ref_proj_matrix, src_proj_matrix

    def _load_img(self, rgb_file):
        img_path = osp.join(self.dataset_root, rgb_file)
        if osp.exists(img_path):
            img = cv2.imread(img_path)
        else:
            img = None
        return img

    def _load_point_cloud(self, file_name):
        points = np.load(file_name)
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def read_calib(self, calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

    def _resize_img(self, img, width, height):
        x_scale = img.shape[1] / width
        y_scale = img.shape[0] / height
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        return img, x_scale, y_scale

    def __getitem__(self, index):
        data_dict = {}

        metadata = self.metadata[index]
        data_dict['seq_id'] = metadata['seq_id']
        data_dict['ref_frame'] = metadata['frame0']
        data_dict['src_frame'] = metadata['frame1']
        data_dict['pcd0_img'] = metadata['pcd0'][:-3] + "png"
        data_dict['pcd1_img'] = metadata['pcd1'][:-3] + "png"
        calib_path = metadata['pcd0'][:-10] + "calib.txt"
        calib = self.read_calib(osp.join(self.dataset_root, calib_path))
        proj_matrix = np.matmul(calib["P2"], calib["Tr"])
        ref_proj_matrix = proj_matrix
        src_proj_matrix = proj_matrix
        ref_points = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd0']))
        src_points = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd1']))
        ref_img = self._load_img(data_dict['pcd0_img'])
        src_img = self._load_img(data_dict['pcd1_img'])
        # resize image
        ref_img, ref_x_scale, ref_y_scale = self._resize_img(ref_img, self.img_width, self.img_height)
        src_img, src_x_scale, src_y_scale = self._resize_img(src_img, self.img_width, self.img_height)
        # normalize image
        ref_img = ref_img / 255.
        src_img = src_img / 255.
        ref_img = (ref_img - self.mean) / self.std
        src_img = (src_img - self.mean) / self.std

        transform = metadata['transform']

        if self.use_augmentation:
            ref_points, src_points, transform, \
             ref_proj_matrix, src_proj_matrix = self._augment_point_cloud(ref_points, src_points,
                                                                          transform, proj_matrix)

        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['ref_img'] = ref_img.astype(np.float32)
        data_dict['src_img'] = src_img.astype(np.float32)
        data_dict['ref_x_scale'] = ref_x_scale
        data_dict['ref_y_scale'] = ref_y_scale
        data_dict['src_x_scale'] = src_x_scale
        data_dict['src_y_scale'] = src_y_scale
        data_dict['ref_proj_matrix'] = ref_proj_matrix.astype(np.float32)
        data_dict['src_proj_matrix'] = src_proj_matrix.astype(np.float32)

        return data_dict

    def __len__(self):
        return len(self.metadata)

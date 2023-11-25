import open3d as o3d
import numpy as np
import argparse


def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)


def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def get_rotation_translation_from_transform(transform: np.ndarray) -> [np.ndarray, np.ndarray]:
    r"""Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def compute_registration_error(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transform (array): ground truth transformation matrix (4, 4)
        est_transform (array): estimated transformation matrix (4, 4)

    Returns:
        rre (float): relative rotation error.
        rte (float): relative translation error.
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    return rre, rte


def apply_transform(points: np.ndarray, transform: np.ndarray, normals=None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',
                        default='./data/0_1.npz')
    parser.add_argument('--num_corr', type=int, default=250)
    parser.add_argument('--Inlier', type=bool, default=False)
    parser.add_argument('--registration', type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':

    cfg = parse_args()
    data_dict = np.load(cfg.file_path)
    ref_points = data_dict['ref_points']
    src_points = data_dict['src_points']
    ref_corr_points = data_dict['ref_corr_points']
    src_corr_points = data_dict['src_corr_points']
    corr_scores = data_dict['corr_scores']
    transform = data_dict['transform']
    estimated_transform = data_dict['estimated_transform']
    delta = [3, 0.0, 0.0]
    radii = [0.005, 0.01, 0.02, 0.04]

    if cfg.num_corr is not None and corr_scores.shape[0] > cfg.num_corr:
        sel_indices = np.argsort(-corr_scores)[: cfg.num_corr]
        ref_corr_points = ref_corr_points[sel_indices]
        src_corr_points = src_corr_points[sel_indices]
        corr_scores = corr_scores[sel_indices]

    src_f = src_corr_points.copy()
    ref_f = ref_corr_points.copy()

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_points)
    pcd_src.paint_uniform_color((249 / 255, 213 / 255, 128 / 255))
    pcd_src.estimate_normals()
    rec_mesh_src = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_src,
                                                                                   o3d.utility.DoubleVector(radii))

    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(ref_points)
    pcd_ref.paint_uniform_color((109 / 255, 139 / 255, 195 / 255))
    pcd_ref.estimate_normals()
    rec_mesh_ref = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_ref,
                                                                                   o3d.utility.DoubleVector(radii))
    if cfg.Inlier:
        pcd_src.translate(delta)
        src_corr_points += delta
        frag = np.concatenate([src_corr_points, ref_corr_points], 0)
        n = src_corr_points.shape[0]
        lines = np.zeros([n, 2])
        lines[:, 0] = np.linspace(0, n - 1, n)
        lines[:, 1] = np.linspace(0, n - 1, n) + n
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(frag),
            lines=o3d.utility.Vector2iVector(lines),
        )
        # 判断哪些是正确的线
        src_f = apply_transform(src_f, transform)
        mask = np.sum(np.sqrt((ref_f - src_f) ** 2), axis=1) < 0.1
        inlier_ratio = np.mean(mask)
        print("IR:", inlier_ratio)
        inlier_colors = np.zeros((n, 3))
        inlier_colors[mask] = [0, 255, 0]
        inlier_colors[~mask] = [255, 0, 0]
        line_set.colors = o3d.utility.Vector3dVector(inlier_colors)

        o3d.visualization.draw_geometries([pcd_src, pcd_ref, line_set])
    elif cfg.registration:
        pcd_src.transform(estimated_transform)
        rre, rte = compute_registration_error(transform, estimated_transform)
        print("RRE:%.3f" % rre)
        print("RTE:%.3f" % rte)
        o3d.visualization.draw_geometries([pcd_src, pcd_ref])
    else:
        o3d.visualization.draw_geometries([pcd_src, pcd_ref])

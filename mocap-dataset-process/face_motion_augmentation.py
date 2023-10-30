import os
import numpy as np

import torch
from torch.nn import functional as F
from tqdm import tqdm
import pandas
import scipy.interpolate

from pytorch3d.transforms import so3_exp_map, so3_log_map


def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def slerp(axisangle_left, axisangle_right, t):
    """Spherical linear interpolation."""
    # https://en.wikipedia.org/wiki/Slerp
    # t: (time - timeleft / (timeright - timeleft)) (0, 1)
    assert (
        axisangle_left.shape == axisangle_right.shape
    ), "axisangle_left and axisangle_right must have the same shape"
    assert (
        axisangle_left.shape[-1] == 3
    ), "axisangle_left and axisangle_right must be axis-angle representations"
    assert (
        t.shape[:-1] == axisangle_left.shape[:-1]
    ), "t must have the same shape as axisangle_left and axisangle_right"

    main_shape = axisangle_left.shape[:-1]
    axisangle_left = axisangle_left.reshape(-1, 3)
    axisangle_right = axisangle_right.reshape(-1, 3)
    t = t.reshape(-1, 1)
    delta_rotation = so3_exp_map(
        so3_log_map(so3_exp_map(-axisangle_left) @ so3_exp_map(axisangle_right)) * t
    )

    return so3_log_map(so3_exp_map(axisangle_left) @ delta_rotation).reshape(
        *main_shape, 3
    )


def slerp_interpolate(motion, new_len):
    motion_len, n_joints, axisangle_dims = motion.shape

    new_t = torch.linspace(0, 1, new_len)
    timeline_idx = new_t * (motion_len - 1)
    timeline_idx_left = torch.floor(timeline_idx).long()
    timeline_idx_right = torch.clamp(timeline_idx_left + 1, max=motion_len - 1)

    motion_left = torch.gather(
        motion, 0, timeline_idx_left[:, None, None].expand(-1, n_joints, axisangle_dims)
    )
    motion_right = torch.gather(
        motion,
        0,
        timeline_idx_right[:, None, None].expand(-1, n_joints, axisangle_dims),
    )
    delta_t = timeline_idx - timeline_idx_left.float()

    new_motion = slerp(
        motion_left,
        motion_right,
        delta_t[:, None, None].expand(-1, n_joints, -1),
    )
    return new_motion


motion_folder = "./datasets/motionx_smplx/motion_data/smplx_322"

matched_files = []
not_matched_files = []

for mocap_dataset in ["humanml", "EgoBody", "GRAB"]:
    mocap_motion_folder = os.path.join(motion_folder, mocap_dataset)

    mocap_motion_files = findAllFile(mocap_motion_folder)
    for mocap_motion_file in tqdm(mocap_motion_files):
        face_motion_file = mocap_motion_file.replace(
            "/motionx_smplx/motion_data/", "/face_motion_data/"
        )
        if mocap_dataset == "humanml":
            filename_without_extension = os.path.splitext(face_motion_file)[0]
            face_motion_file = face_motion_file.replace(
                filename_without_extension, f"{filename_without_extension}_clip0000"
            )
        motion = np.load(mocap_motion_file)
        if not os.path.exists(face_motion_file):
            not_matched_files.append(mocap_motion_file)
            continue
        face_motion = np.load(face_motion_file)

        motion_length, face_motion_length = motion.shape[0], face_motion.shape[0]
        if motion_length != face_motion_length:
            face_motion = torch.from_numpy(face_motion)
            n_frames, n_dims = face_motion.shape
            n_joints = n_dims // 3
            face_motion = face_motion.reshape(n_frames, n_joints, 3)
            face_motion = slerp_interpolate(face_motion, motion_length)
            face_motion = face_motion.reshape(motion_length, -1).numpy()
        else:
            (
                motion[:, 66 + 90 : 66 + 93],
                motion[:, 159 : 159 + 50],
                motion[:, 209 : 209 + 100],
            ) = (face_motion[:, :3], face_motion[:, 3 : 3 + 50], face_motion[:, 53:153])
        np.save(mocap_motion_file, motion)

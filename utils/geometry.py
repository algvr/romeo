import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(1, project_root)

# Copyright (c) Facebook, Inc. and its affiliates.
import math

import torch
from torch.nn import functional as F


def rot6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural
    Networks", CVPR 2019

    Args:
        rot_6d (B x 6): Batch of 6D Rotation representation.

    Returns:
        Rotation matrices (B x 3 x 3).
    """
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(rotmat):
    """
    Convert rotation matrix to 6D rotation representation.

    Args:
        rotmat (B x 3 x 3): Batch of rotation matrices.

    Returns:
        6D Rotations (B x 3 x 2).
    """
    return rotmat.view(-1, 3, 3)[:, :, :2]


def combine_verts(verts_list):
    all_verts_list = [v.reshape(1, -1, 3) for v in verts_list]
    verts_combined = torch.cat(all_verts_list, 1)
    return verts_combined


def center_vertices(vertices, faces, flip_y=True):
    """
    Centroid-align vertices.

    Args:
        vertices (V x 3): Vertices.
        faces (F x 3): Faces.
        flip_y (bool): If True, flips y verts to keep with image coordinates convention.

    Returns:
        vertices, faces
    """
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    if flip_y:
        vertices[:, 1] *= -1
        faces = faces[:, [2, 1, 0]]
    return vertices, faces


def compute_dist_z(verts1, verts2):
    """
    Computes distance between sets of vertices only in Z-direction.

    Args:
        verts1 (V x 3).
        verts2 (V x 3).

    Returns:
        tensor
    """
    a = verts1[:, 2].min()
    b = verts1[:, 2].max()
    c = verts2[:, 2].min()
    d = verts2[:, 2].max()
    if d >= a and b >= c:
        return 0.0
    return torch.min(torch.abs(c - b), torch.abs(a - d))


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


# pytorch3d transforms
# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#euler_angles_to_matrix
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def compute_random_rotations(B=10, upright=False):
    """
    Randomly samples rotation matrices.

    Args:
        B (int): Batch size.
        upright (bool): If True, samples rotations that are mostly upright. Otherwise,
            samples uniformly from rotation space.

    Returns:
        rotation_matrices (B x 3 x 3).
    """
    if upright:
        a1 = torch.FloatTensor(B, 1).uniform_(0, 2 * math.pi)
        a2 = torch.FloatTensor(B, 1).uniform_(-math.pi / 6, math.pi / 6)
        a3 = torch.FloatTensor(B, 1).uniform_(-math.pi / 12, math.pi / 12)

        angles = torch.cat((a1, a2, a3), 1).cuda()
        # angles = torch.cat((a1, a2, a3), 1)
        rotation_matrices = euler_angles_to_matrix(angles, "YXZ")
    else:
        # Reference: J Avro. "Fast Random Rotation Matrices." (1992)
        x1, x2, x3 = torch.split(torch.rand(3 * B).cuda(), B)
        # x1, x2, x3 = torch.split(torch.rand(3 * B), B)
        tau = 2 * math.pi
        R = torch.stack(
            (  # B x 3 x 3
                torch.stack(
                    (torch.cos(tau * x1), torch.sin(tau * x1), torch.zeros_like(x1)), 1
                ),
                torch.stack(
                    (-torch.sin(tau * x1), torch.cos(tau * x1), torch.zeros_like(x1)), 1
                ),
                torch.stack(
                    (torch.zeros_like(x1), torch.zeros_like(x1), torch.ones_like(x1)), 1
                ),
            ),
            1,
        )
        v = torch.stack(
            (  # B x 3
                torch.cos(tau * x2) * torch.sqrt(x3),
                torch.sin(tau * x2) * torch.sqrt(x3),
                torch.sqrt(1 - x3),
            ),
            1,
        )
        identity = torch.eye(3).repeat(B, 1, 1).cuda()
        # identity = torch.eye(3).repeat(B, 1, 1)
        H = identity - 2 * v.unsqueeze(2) * v.unsqueeze(1)
        rotation_matrices = -torch.matmul(H, R)
    return rotation_matrices


def compute_random_rotations_optuna(study, trial_numbers, B=10, upright=False):
    """
    Randomly samples rotation matrices.

    Args:
        B (int): Batch size.
        upright (bool): If True, samples rotations that are mostly upright. Otherwise,
            samples uniformly from rotation space.

    Returns:
        rotation_matrices (B x 3 x 3).
    """
    if upright:
        a1_batch = []
        a2_batch = []
        a3_batch = []
        for _ in range(B):
            trial = study.ask()
            trial_numbers.append(trial.number)
            a1_batch.append([trial.suggest_float("a1", 0, 2*math.pi)])
            a2_batch.append([trial.suggest_float("a2", -math.pi / 6, math.pi / 6)])
            a3_batch.append([trial.suggest_float("a3", -math.pi / 12, math.pi / 12)])

        a1 = torch.tensor(a1_batch)
        a2 = torch.tensor(a2_batch)
        a3 = torch.tensor(a3_batch)

        angles = torch.cat((a1, a2, a3), 1).cuda()
        # angles = torch.cat((a1, a2, a3), 1)
        rotation_matrices = euler_angles_to_matrix(angles, "YXZ")
    else:
        # Reference: J Avro. "Fast Random Rotation Matrices." (1992)

        x1_batch = []
        x2_batch = []
        x3_batch = []
        for _ in range(B):
            trial = study.ask()
            trial_numbers.append(trial.number)
            x1_batch.append(trial.suggest_float("x1", 0, 1))
            x2_batch.append(trial.suggest_float("x2", 0, 1))
            x3_batch.append(trial.suggest_float("x3", 0, 1))

        x1 = torch.tensor(x1_batch).cuda()
        x2 = torch.tensor(x2_batch).cuda()
        x3 = torch.tensor(x3_batch).cuda()
        
        # x1, x2, x3 = torch.split(torch.rand(3 * B), B)
        tau = 2 * math.pi
        R = torch.stack(
            (  # B x 3 x 3
                torch.stack(
                    (torch.cos(tau * x1), torch.sin(tau * x1), torch.zeros_like(x1)), 1
                ),
                torch.stack(
                    (-torch.sin(tau * x1), torch.cos(tau * x1), torch.zeros_like(x1)), 1
                ),
                torch.stack(
                    (torch.zeros_like(x1), torch.zeros_like(x1), torch.ones_like(x1)), 1
                ),
            ),
            1,
        )
        v = torch.stack(
            (  # B x 3
                torch.cos(tau * x2) * torch.sqrt(x3),
                torch.sin(tau * x2) * torch.sqrt(x3),
                torch.sqrt(1 - x3),
            ),
            1,
        )
        identity = torch.eye(3).repeat(B, 1, 1).cuda()
        # identity = torch.eye(3).repeat(B, 1, 1)
        H = identity - 2 * v.unsqueeze(2) * v.unsqueeze(1)
        rotation_matrices = -torch.matmul(H, R)
    return rotation_matrices


if __name__ == "__main__":
    r = compute_random_rotations(1, upright=True)
    print(r)

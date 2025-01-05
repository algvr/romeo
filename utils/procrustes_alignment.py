import os
import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(1, project_root)

import numpy as np
import torch

from utils.obj_mesh_processing import load_obj


def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat, scale, R, t


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    scale = []
    R = []
    t = []
    for i in range(S1.shape[0]):
        S1_hat[i], _s, _r, _t = compute_similarity_transform(S1[i], S2[i])
        scale.append(_s)
        R.append(_r)
        t.append(_t)
    return S1_hat, scale, R, t


def eval(pred_path, obj_gt_path, person_gt_path):
    recon = load_obj(pred_path)[0]
    obj_vert_num = recon.shape[0] - 6890

    from psbody.mesh import Mesh
    m = Mesh()
    m.load_from_file(obj_gt_path)
    obj_gt = m.v 
    m = Mesh()
    m.load_from_file(person_gt_path)
    person_gt = m.v 

    obj_gt = torch.tensor(obj_gt).float().unsqueeze(0)
    person_gt = torch.tensor(person_gt).float().unsqueeze(0)

    recon = torch.tensor(recon).float().unsqueeze(0)
    
    body_aligned, scale, R, t = compute_similarity_transform_batch(recon[:, obj_vert_num:].numpy(), person_gt.numpy()) 
    obj_aligned = (scale[0] * R[0].dot(recon[0, :obj_vert_num].numpy().T) + t[0]).T

    return body_aligned, obj_aligned

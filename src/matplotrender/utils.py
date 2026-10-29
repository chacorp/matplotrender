"""
Shared utilities used by both matplotrender.py and pyrender.py.
"""

import numpy as np
import subprocess
import torch


# ── rotation helpers ──────────────────────────────────────────────────────────

def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[ 1, 0, 0, 0],
                     [ 0, c,-s, 0],
                     [ 0, s, c, 0],
                     [ 0, 0, 0, 1]], dtype=float)

def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]], dtype=float)

def zrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return np.array([[ c,-s, 0, 0],
                     [ s, c, 0, 0],
                     [ 0, 0, 1, 0],
                     [ 0, 0, 0, 1]], dtype=float)

def get_rotation_matrix(rotations):
    """
    Args:
        rotations: List[float, float, float]
            list of x, y, z rotations in euler angle
    Returns:
        matrix : np.ndarray
            4x4 rotation matrix
    """
    xrot, yrot, zrot = rotations
    return yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)


# ── mesh geometry ─────────────────────────────────────────────────────────────

def calc_face_norm(vertices, faces, mode='faces'):
    """
    Args
    -------
        vertices : np.ndarray
            vertices
        faces : np.ndarray
            face indices
        mode : str
            whether to calculate normals on 'faces' or 'vertices'

    Returns
    -------
        norm or norm_v (np.ndarray):
            normals on 'faces' or 'vertices'
    """
    fv = vertices[faces]
    span = fv[:, 1:, :] - fv[:, :1, :]
    norm = np.cross(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[:, np.newaxis] + 1e-12)

    if mode == 'faces':
        return norm

    vertex_normals = np.zeros(vertices.shape, dtype=np.float64)
    np.add.at(vertex_normals, faces[:, 0], norm)
    np.add.at(vertex_normals, faces[:, 1], norm)
    np.add.at(vertex_normals, faces[:, 2], norm)
    norm_v = vertex_normals / (np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis] + 1e-12)
    return norm_v


def fix_triangle_widning(vertices, faces):
    """Flip triangle winding so the mesh has a consistent, outward-facing orientation.

    Face-normal based shading and backface culling in this module assume
    CCW (outward-facing, positive signed volume) winding. Some source
    meshes are exported with the opposite winding, which flips every face
    normal and makes the renderer cull the visible outer surface while
    keeping hidden interior geometry (e.g. eye sockets, inner mouth)
    instead. This detects that case from the mesh's signed volume and
    reverses every face's winding if needed.

    Args
    -------
        vertices : np.ndarray
            [V, 3] vertex positions
        faces : np.ndarray
            [F, 3] triangle indices

    Returns
    -------
        faces (np.ndarray):
            [F, 3] triangle indices with consistent outward winding
    """
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    signed_volume = np.sum(np.einsum('ij,ij->i', v0, np.cross(v1, v2)))
    if signed_volume < 0:
        faces = faces[:, [0, 2, 1]]
    return faces


# ── data helpers ──────────────────────────────────────────────────────────────

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _compute_face_diff(src_v, src_f, anc_v):
    """Per-face mean L2 distance between src and anc vertices.
    Same vertex count → direct subtraction.
    Different count   → nearest-neighbour per src vertex.
    """
    if src_v.shape[0] == anc_v.shape[0]:
        vert_dist = np.linalg.norm(src_v - anc_v, axis=-1)
    else:
        # (V_src, V_anc, 3) — fine for meshes up to ~10k verts
        vert_dist = np.linalg.norm(
            src_v[:, np.newaxis, :] - anc_v[np.newaxis, :, :], axis=-1
        ).min(axis=1)
    return vert_dist[src_f].mean(axis=1)

def _mux_audio(audio_fn, tmp_path, out_path):
    print("[INFO] mux audio and video")
    cmd = f"ffmpeg -y -i {audio_fn} -i {tmp_path} -c:v copy -c:a aac {out_path}"
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.call(f"rm -f {tmp_path}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

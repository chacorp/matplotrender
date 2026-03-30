"""
pyrender-based mesh rendering — same public API as matplotrender.py.

For headless (off-screen) rendering, set this BEFORE importing:

    import os
    os.environ["PYOPENGL_PLATFORM"] = "egl"    # Linux + NVIDIA EGL
    os.environ["PYOPENGL_PLATFORM"] = "osmesa" # CPU software rendering

On Windows with a display attached, no variable is needed.

Public functions (same signatures as matplotrender.py):
    plot_mesh_image
    plot_mesh_video
    plot_image_array_diff3
    render_video_mesh_diff
    render_video_mesh_diffs
    plot_mesh_gouraud
"""

import os
import sys as _sys
import importlib.util as _ilu
import sysconfig as _sc
import numpy as np
import trimesh

# This file is named pyrender.py, which conflicts with the installed pyrender package.
# Load the real package directly from site-packages to avoid the name collision.
def _load_real_pyrender():
    site = _sc.get_paths()['purelib']
    pkg_dir = os.path.join(site, 'pyrender')
    init = os.path.join(pkg_dir, '__init__.py')
    if not os.path.exists(init):
        raise ImportError(f"pyrender package not found at {init}. Run: pip install pyrender")
    spec = _ilu.spec_from_file_location(
        'pyrender', init,
        submodule_search_locations=[pkg_dir])
    mod = _ilu.module_from_spec(spec)
    mod.__path__ = [pkg_dir]
    mod.__package__ = 'pyrender'
    _sys.modules['pyrender'] = mod   # must be set BEFORE exec so relative imports resolve
    spec.loader.exec_module(mod)
    return mod

_pyrender = _load_real_pyrender()
del _load_real_pyrender

import imageio
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Union
import subprocess
import torch


# ── math helpers (shared with matplotrender) ──────────────────────────────────

def _xrotate(theta):
    t = np.pi * theta / 180; c, s = np.cos(t), np.sin(t)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=float)

def _yrotate(theta):
    t = np.pi * theta / 180; c, s = np.cos(t), np.sin(t)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=float)

def _zrotate(theta):
    t = np.pi * theta / 180; c, s = np.cos(t), np.sin(t)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)

def get_rotation_matrix(rotations):
    xrot, yrot, zrot = rotations
    return _yrotate(yrot) @ _xrotate(xrot) @ _zrotate(zrot)

def calc_face_norm(vertices, faces, mode='faces'):
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
    return vertex_normals / (np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis] + 1e-12)

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _compute_face_diff(src_v, src_f, anc_v):
    if src_v.shape[0] == anc_v.shape[0]:
        vert_dist = np.linalg.norm(src_v - anc_v, axis=-1)
    else:
        vert_dist = np.linalg.norm(
            src_v[:, np.newaxis, :] - anc_v[np.newaxis, :, :], axis=-1
        ).min(axis=1)
    return vert_dist[src_f].mean(axis=1)

def _mux_audio(audio_fn, tmp_path, out_path):
    print("[INFO] mux audio and video")
    cmd = f"ffmpeg -y -i {audio_fn} -i {tmp_path} -c:v copy -c:a aac {out_path}"
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.call(f"rm -f {tmp_path}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# ── pyrender helpers ──────────────────────────────────────────────────────────

def _normalize_mesh(V):
    """Center mesh and scale so longest axis spans [-1, 1]."""
    V = V - (V.max(0) + V.min(0)) * 0.5
    span = max(V.max(0) - V.min(0))
    if span > 0:
        V = V / span
    return V

def _face_colors_to_vertex(face_colors, F, num_verts):
    """Average per-face colors onto vertices."""
    nc = face_colors.shape[1]
    vc = np.zeros((num_verts, nc))
    np.add.at(vc, F[:, 0], face_colors)
    np.add.at(vc, F[:, 1], face_colors)
    np.add.at(vc, F[:, 2], face_colors)
    counts = np.zeros(num_verts)
    np.add.at(counts, F.ravel(), 1)
    return vc / np.maximum(counts[:, np.newaxis], 1)

def _build_scene(bg_black):
    """Scene with full ambient light so pre-computed vertex colors are exact."""
    bg = np.array([0.0, 0.0, 0.0, 1.0]) if bg_black else np.array([1.0, 1.0, 1.0, 1.0])
    return _pyrender.Scene(bg_color=bg, ambient_light=np.array([1.0, 1.0, 1.0, 1.0]))

def _camera_pose():
    """Camera at (0, 0, 2) looking toward origin down -Z."""
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]], dtype=float)

def _add_camera(scene, orth_view=True, fovy=55):
    cam = (_pyrender.OrthographicCamera(xmag=1.0, ymag=1.0) if orth_view
           else _pyrender.PerspectiveCamera(yfov=np.radians(fovy)))
    scene.add(cam, pose=_camera_pose())

def _add_mesh_node(scene, V, F, vertex_colors, rot_mat, smooth=True):
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    vc_uint8 = np.clip(vertex_colors * 255, 0, 255).astype(np.uint8)
    if vc_uint8.shape[1] == 3:
        vc_uint8 = np.hstack([vc_uint8, np.full((len(vc_uint8), 1), 255, dtype=np.uint8)])
    tm.visual.vertex_colors = vc_uint8
    mesh = _pyrender.Mesh.from_trimesh(tm, smooth=smooth)
    pose = np.eye(4)
    pose[:3, :3] = rot_mat[:3, :3]
    scene.add(mesh, pose=pose)

def _render_scene(scene, width, height, wireframe=False):
    renderer = _pyrender.OffscreenRenderer(width, height)
    flags = _pyrender.RenderFlags.ALL_WIREFRAME if wireframe else _pyrender.RenderFlags.NONE
    color, _ = renderer.render(scene, flags=flags)
    renderer.delete()
    return color  # (H, W, 3) uint8

def _get_rot(rot_list, idx):
    return rot_list[idx] if rot_list else [0, 0, 0]

def _show_or_save_img(combined, save, savepath, show):
    img = Image.fromarray(combined)
    if save and savepath:
        os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
        img.save(savepath)
    if show:
        img.show()


# ── vertex-color computation per mode ────────────────────────────────────────

def _vc_shade(V, F, rot_mat, light_dir):
    ld = np.array(light_dir, dtype=float)
    ld = ld / (np.linalg.norm(ld) + 1e-12)
    model_rot = rot_mat[:3, :3]
    face_normals = calc_face_norm(V, F)
    face_normals_view = (model_rot @ face_normals.T).T
    shading = np.clip(face_normals_view @ ld, 0, 1)[:, np.newaxis].repeat(3, axis=-1)
    shading = shading * 0.7 + 0.2
    return _face_colors_to_vertex(shading, F, len(V))

def _vc_normal(V, F, rot_mat):
    model_rot = rot_mat[:3, :3]
    face_normals = calc_face_norm(V, F)
    face_normals_view = (model_rot @ face_normals.T).T
    colors = np.clip(face_normals_view * 0.5 + 0.5, 0, 1)
    return _face_colors_to_vertex(colors, F, len(V))

def _vc_depth(V, F, rot_mat):
    # simple depth-based gray (no MVP needed; approx by rotated z)
    Vr = (rot_mat[:3, :3] @ V.T).T
    face_z = Vr[F, 2].mean(axis=1)
    zmin, zmax = face_z.min(), face_z.max()
    z_norm = (face_z - zmin) / (zmax - zmin + 1e-12)
    gray = z_norm[:, np.newaxis].repeat(3, axis=-1) * 0.6 + 0.2
    return _face_colors_to_vertex(gray, F, len(V))

def _vc_shade_with_diff(V, F, rot_mat, light_dir, diff_norm, c_map='YlOrRd'):
    """Blend face-level shade colors with a diff colormap, return vertex colors."""
    import matplotlib.pyplot as plt
    ld = np.array(light_dir, dtype=float)
    ld = ld / (np.linalg.norm(ld) + 1e-12)
    model_rot = rot_mat[:3, :3]
    # compute shade at face level
    face_normals = calc_face_norm(V, F)
    normals_view = (model_rot @ face_normals.T).T
    shading = np.clip(normals_view @ ld, 0, 1)[:, np.newaxis].repeat(3, axis=-1)
    # blend with diff colormap (both are face-level here)
    face_cmap = plt.get_cmap(c_map)(diff_norm)[..., :3]   # (F, 3)
    mask = diff_norm[:, np.newaxis]                         # (F, 1)
    blended = shading * (1 - mask) + face_cmap * mask
    blended = np.clip(blended * 0.7 + 0.2, 0, 1)
    return _face_colors_to_vertex(blended, F, len(V))


# ── single-frame render ───────────────────────────────────────────────────────

def _render_panel(V, F, rot, mode, norm, light_dir, bg_black, img_size,
                  vertex_colors_override=None, orth_view=True, fovy=55):
    """Render one mesh panel; return (img_size, img_size, 3) uint8."""
    V = _to_numpy(V).astype(float)
    F = _to_numpy(F).astype(int)
    V = _normalize_mesh(V)
    rot_mat = get_rotation_matrix(rot)

    if vertex_colors_override is not None:
        vc = vertex_colors_override
    elif mode == 'shade':
        vc = _vc_shade(V, F, rot_mat, light_dir)
    elif mode == 'normal':
        vc = _vc_normal(V, F, rot_mat)
    else:  # 'mesh' — depth gray + wireframe overlay below
        vc = _vc_depth(V, F, rot_mat)

    wireframe = (mode == 'mesh')
    scene = _build_scene(bg_black)
    _add_mesh_node(scene, V, F, vc, rot_mat, smooth=(mode not in ('mesh',)))
    _add_camera(scene, orth_view, fovy)
    return _render_scene(scene, img_size, img_size, wireframe=wireframe)


# ── public API ────────────────────────────────────────────────────────────────

def plot_mesh_image(
        Vs,
        Fs,
        rot_list=None,
        size=6,
        norm=False,
        mode='mesh',
        z_pos=-5,            # kept for API compatibility (ignored)
        custom_perspective=None,  # kept for API compatibility (ignored)
        linewidth=1,         # kept for API compatibility (ignored)
        linestyle='solid',   # kept for API compatibility (ignored)
        light_dir=np.array([0, 0, 1]),
        bg_black=False,
        savedir='.',
        name='000',
        save=False,
    ):
    """
    Render static images of one or more meshes side by side using pyrender.

    Args:
        Vs (list): list of vertex arrays [V, V, ...]
        Fs (list): list of face index arrays [F, F, ...]
        rot_list (list): euler rotations [[x,y,z], ...] per mesh
        size (int): panel size in 100-px units (size=6 → 600×600 per panel)
        norm (bool): kept for API compatibility (vertices are always normalized)
        mode (str): 'mesh' | 'shade' | 'normal'
        light_dir (np.array): light direction for shade mode
        bg_black (bool): black background
        savedir (str): directory for saved image
        name (str): filename stem
        save (bool): save to disk when True, show when False
    """
    img_size = size * 100
    panels = []
    for idx, (V, F) in enumerate(zip(Vs, Fs)):
        rot = _get_rot(rot_list, idx)
        panel = _render_panel(V, F, rot, mode, norm, light_dir, bg_black, img_size)
        panels.append(panel)

    combined = np.concatenate(panels, axis=1)
    _show_or_save_img(combined, save, f'{savedir}/{name}.png', show=not save)


def plot_mesh_gouraud(
        Vs,
        Fs,
        Cs=None,
        rot_list=None,
        size=6,
        norm=False,
        mode='shade',
        threshold=0.01,     # kept for API compatibility
        seg_divide=False,   # kept for API compatibility
        is_diff=False,
        is_color=False,
        diff_base=None,
        diff_revert=False,
        backface_culling=False,  # kept for API compatibility
        depth_sorting=True,      # kept for API compatibility
        linewidth=1,
        linestyle='solid',
        light_dir=np.array([0, 0, 1]),
        light_at_frontface=True,
        view_dir=np.array([0, 0, 1]),
        mesh_scale=1.0,
        mesh_trans=np.array([0, 0, 0]),
        c_map='nipy_spectral',
        blend=0.5,
        seg_only=-1,        # kept for API compatibility
        orth_view=True,
        fovy=55,
        bg_black=True,
        logdir='.',
        name='000',
        save=False,
        show=True,
    ):
    """
    Render meshes with smooth (Gouraud-equivalent) shading using pyrender.

    Args:
        Vs (list of np.ndarray): vertex arrays
        Fs (list of np.ndarray): face index arrays
        Cs (list of np.ndarray | None): per-vertex color/segment arrays
        rot_list (list): euler rotations [[x,y,z], ...] per mesh
        mode (str): 'shade' | 'normal'
        is_diff (bool): color by vertex displacement from diff_base
        is_color (bool): use Cs as raw vertex RGB
        diff_base (np.ndarray | None): reference vertex array for diff
        light_dir (np.array): light direction
        bg_black (bool): black background
    """
    import matplotlib.pyplot as plt

    if is_diff and diff_base is None:
        raise ValueError('diff_base must be provided when is_diff=True')

    num_meshes = len(Vs)
    if Cs is None:
        Cs = [None] * num_meshes

    mesh_trans_np = _to_numpy(mesh_trans)
    light_dir_np = np.array(light_dir, dtype=float)
    light_dir_np = light_dir_np / (np.linalg.norm(light_dir_np) + 1e-12)

    if is_diff:
        Vs_np = np.array([_to_numpy(v) for v in Vs])
        diff_base_np = _to_numpy(diff_base)
        C_global = np.linalg.norm(np.abs(Vs_np - diff_base_np), axis=-1)
        C_global = (C_global - C_global.min()) / (C_global.max() - C_global.min() + 1e-12)
        if diff_revert:
            C_global = 1.0 - C_global

    img_size = size * 100
    panels = []

    for idx, (V, F, C) in enumerate(zip(Vs, Fs, Cs)):
        V = _to_numpy(V).astype(float)
        F = _to_numpy(F).astype(int)
        C = _to_numpy(C)

        V = V * mesh_scale + mesh_trans_np
        V = _normalize_mesh(V)

        rot = _get_rot(rot_list, idx)
        rot_mat = get_rotation_matrix(rot)

        # Base shading
        if mode == 'shade' or mode == 'gouraud':
            base_vc = _vc_shade(V, F, rot_mat, light_dir_np)
        else:
            base_vc = _vc_normal(V, F, rot_mat)

        # Override / blend vertex colors
        if is_diff:
            diff_vc = C_global[idx]  # per-vertex diff [0,1]
            cmap_vc = plt.get_cmap('YlOrRd')(diff_vc)[..., :3]
            mask = diff_vc[:, np.newaxis]
            base_vc = base_vc * (1 - mask) + cmap_vc * mask
            base_vc = np.clip(base_vc, 0, 1)
        elif is_color and C is not None:
            base_vc = np.clip(C, 0, 1)
        elif C is not None:
            # segment logits → color via c_map
            import torch as _torch
            if isinstance(C, _torch.Tensor):
                C = C.detach().cpu().numpy()
            len_seg = C.shape[-1]
            exp_c = np.exp(C - C.max(-1, keepdims=True))
            seg_label = (exp_c / exp_c.sum(-1, keepdims=True)).argmax(-1).astype(float)
            seg_vc = plt.get_cmap(c_map)(seg_label / len_seg)[..., :3]
            base_vc = base_vc * (1 - blend) + seg_vc * blend
            base_vc = np.clip(base_vc, 0, 1)

        scene = _build_scene(bg_black)
        _add_mesh_node(scene, V, F, base_vc, rot_mat, smooth=True)
        _add_camera(scene, orth_view, fovy)
        panels.append(_render_scene(scene, img_size, img_size))

    combined = np.concatenate(panels, axis=1)
    _show_or_save_img(combined, save, f'{logdir}/{name}.png', show=show)


def plot_image_array_diff3(
        Vs,
        Fs,
        D,
        rot=(0, 0, 0),
        size=6,
        norm=False,
        linewidth=1,
        linestyle='solid',
        light_dir=np.array([0, 0, 1]),
        bg_black=True,
        threshold=None,
        logdir='.',
        name='000',
        save=False,
        draw_base=True,
        c_map='YlOrRd',
    ):
    """
    Render mesh panels coloured by per-face L2 distance from anchor D.

    Layout: [ base (D) | Vs[0] | Vs[1] | ... ]
    """
    import matplotlib.pyplot as plt

    img_size = size * 100
    rot_mat = get_rotation_matrix(list(rot) if rot is not None else [0, 0, 0])
    light_dir_np = np.array(light_dir, dtype=float)

    D_np = _to_numpy(D).astype(float)

    panels = []

    # base panel
    if draw_base:
        V_d = _normalize_mesh(D_np.copy())
        vc = _vc_shade(V_d, _to_numpy(Fs[0]).astype(int), rot_mat, light_dir_np)
        scene = _build_scene(bg_black)
        _add_mesh_node(scene, V_d, _to_numpy(Fs[0]).astype(int), vc, rot_mat)
        _add_camera(scene)
        panels.append(_render_scene(scene, img_size, img_size))

    # compute global diff range
    D_diff_all = []
    for V, F in zip(Vs, Fs):
        V_np = _to_numpy(V).astype(float)
        F_np = _to_numpy(F).astype(int)
        d = np.abs(D_np - V_np)[F_np]
        d = np.linalg.norm(d, axis=-1).mean(axis=-1)
        D_diff_all.append(d)
    D_diff_all = np.array(D_diff_all)
    if threshold is not None:
        D_diff_all = np.clip(D_diff_all, 0, threshold)
    diff_min, diff_max = D_diff_all.min(), D_diff_all.max()

    for idx, (V, F) in enumerate(zip(Vs, Fs)):
        V_np = _normalize_mesh(_to_numpy(V).astype(float))
        F_np = _to_numpy(F).astype(int)
        diff = D_diff_all[idx]
        diff_norm = (diff - diff_min) / (diff_max - diff_min + 1e-12) if diff_max > 0 else diff
        vc = _vc_shade_with_diff(V_np, F_np, rot_mat, light_dir_np, diff_norm, c_map)
        scene = _build_scene(bg_black)
        _add_mesh_node(scene, V_np, F_np, vc, rot_mat)
        _add_camera(scene)
        panels.append(_render_scene(scene, img_size, img_size))

    combined = np.concatenate(panels, axis=1)
    _show_or_save_img(combined, save, f'{logdir}/{name}.png', show=not save)


def plot_mesh_video(
        Vs,
        Fs,
        D=None,
        rot_list=None,
        size: int = 6,
        norm: bool = False,
        linewidth: float = 1,
        light_dir=np.array([0, 0, 1]),
        bg_black: bool = False,
        threshold: float = None,
        c_map: str = 'YlOrRd',
        fps: int = 30,
        savedir: Optional[str] = 'tmp',
        savename: Optional[str] = 'test',
        audio_dir: Optional[str] = None,
        debug=False,
    ):
    """
    Render a mesh sequence as a video using pyrender.

    Args:
        Vs (list of np.ndarray): each element is (T, V, 3) vertex sequence
        Fs (list of np.ndarray): face arrays; Fs[0] is used for all frames
        D (np.ndarray | None): anchor (T, V, 3) for diff coloring
        rot_list (list): per-frame euler rotations [[x,y,z], ...]
        fps (int): output video frame rate
        savedir (str): output directory
        savename (str): output filename stem
        audio_dir (str | None): path to audio file to mux in
    """
    import matplotlib.pyplot as plt

    Vs = np.array([_to_numpy(v) for v in Vs])
    F = _to_numpy(Fs[0]).astype(int)
    num_frames = len(Vs[0])
    img_size = size * 100
    light_dir_np = np.array(light_dir, dtype=float)

    os.makedirs(savedir, exist_ok=True)
    anim_name = f'{savedir}/{savename}.mp4'
    tmp_path = f'{savedir}/_tmp_.mp4' if audio_dir is not None else anim_name

    frames = []
    for frame_idx in tqdm(range(num_frames), desc="rendering"):
        rot = _get_rot(rot_list, frame_idx)
        rot_mat = get_rotation_matrix(rot)
        panels = []

        if D is not None:
            # anchor panel
            V_anc = _normalize_mesh(_to_numpy(D[frame_idx]).astype(float))
            vc = _vc_shade(V_anc, F, rot_mat, light_dir_np)
            scene = _build_scene(bg_black)
            _add_mesh_node(scene, V_anc, F, vc, rot_mat)
            _add_camera(scene)
            panels.append(_render_scene(scene, img_size, img_size))

        for v_seq in Vs:
            V = _normalize_mesh(_to_numpy(v_seq[frame_idx]).astype(float))

            if D is not None:
                V_anc = _to_numpy(D[frame_idx]).astype(float)
                diff = _compute_face_diff(V, F, _normalize_mesh(V_anc))
                if threshold is not None:
                    diff = np.clip(diff, 0, threshold)
                diff_max = diff.max()
                diff_norm = diff / (diff_max + 1e-12) if diff_max > 0 else diff
                vc = _vc_shade_with_diff(V, F, rot_mat, light_dir_np, diff_norm, c_map)
            else:
                vc = _vc_shade(V, F, rot_mat, light_dir_np)

            scene = _build_scene(bg_black)
            _add_mesh_node(scene, V, F, vc, rot_mat)
            _add_camera(scene)
            panels.append(_render_scene(scene, img_size, img_size))

        frames.append(np.concatenate(panels, axis=1))

    imageio.mimwrite(tmp_path, frames, fps=fps, codec='libx264', quality=8)
    print(f"saved as: {anim_name}")

    if audio_dir is not None:
        _mux_audio(audio_dir, tmp_path, anim_name)


def render_video_mesh_diff(
        Vs,
        Fs,
        D=None,
        rot_list=None,
        size=6,
        norm=False,
        linewidth=1,
        light_dir=np.array([0, 0, 1]),
        bg_black=True,
        threshold=None,
        c_map='YlOrRd',
        savedir=None,
        savename='temp',
        audio_fn=None,
        fps=30,
    ):
    """
    Render a diff-colored mesh video using pyrender.

    Layout per frame: [ anchor (D) | Vs[0] | Vs[1] | ... ]
    If D is None, Vs[0][0] is repeated as the anchor.
    """
    if D is None:
        Vs0 = _to_numpy(Vs[0])
        D = np.repeat(Vs0[[0]], len(Vs0), axis=0)

    plot_mesh_video(
        Vs=Vs,
        Fs=Fs,
        D=D,
        rot_list=rot_list,
        size=size,
        norm=norm,
        light_dir=light_dir,
        bg_black=bg_black,
        threshold=threshold,
        c_map=c_map,
        fps=fps,
        savedir=savedir if savedir is not None else 'tmp',
        savename=savename,
        audio_dir=audio_fn,
    )


def render_video_mesh_diffs(
        src_vs,
        src_fs,
        anc_vs,
        anc_fs,
        rot_list=None,
        size=6,
        norm=False,
        linewidth=1,
        light_dir=np.array([0, 0, 1]),
        bg_black=False,
        threshold=None,
        c_map='YlOrRd',
        fps=30,
        savedir=None,
        savename='temp',
        audio_fn=None,
    ):
    """
    Compare src sequences against anchor anc_vs frame by frame.

    Layout per frame: [ anchor | src_0 | src_1 | ... ]
    Colormap is normalized globally across all frames.

    Args:
        src_vs (list of np.ndarray): list of (T, V, 3) source sequences
        src_fs (list of np.ndarray): face arrays for each src
        anc_vs (np.ndarray): (T, V, 3) anchor sequence
        anc_fs (np.ndarray): anchor face array
    """
    if not isinstance(src_vs, list):
        src_vs = [src_vs]
    if not isinstance(src_fs, list):
        src_fs = [src_fs]

    src_vs  = [np.asarray(_to_numpy(v)) for v in src_vs]
    src_fs  = [np.asarray(_to_numpy(f)).astype(int) for f in src_fs]
    anc_vs  = np.asarray(_to_numpy(anc_vs))
    anc_f   = np.asarray(_to_numpy(anc_fs)).astype(int)

    num_frames = len(anc_vs)
    img_size = size * 100
    light_dir_np = np.array(light_dir, dtype=float)

    # global diff_max for consistent colormap
    all_diffs = [
        _compute_face_diff(src_vs[si][t], src_fs[si], anc_vs[t])
        for t in range(num_frames)
        for si in range(len(src_vs))
    ]
    all_diffs_cat = np.concatenate(all_diffs)
    if threshold is not None:
        all_diffs_cat = np.clip(all_diffs_cat, 0, threshold)
    diff_max = float(all_diffs_cat.max())

    os.makedirs(savedir or 'tmp', exist_ok=True)
    out_dir = savedir or 'tmp'
    anim_name = f'{out_dir}/{savename}.mp4'
    tmp_path = f'{out_dir}/_tmp_.mp4' if audio_fn is not None else anim_name

    frames = []
    for frame_idx in tqdm(range(num_frames), desc="rendering"):
        rot = _get_rot(rot_list, frame_idx)
        rot_mat = get_rotation_matrix(rot)
        panels = []

        # anchor panel
        V_anc = _normalize_mesh(anc_vs[frame_idx].astype(float))
        vc = _vc_shade(V_anc, anc_f, rot_mat, light_dir_np)
        scene = _build_scene(bg_black)
        _add_mesh_node(scene, V_anc, anc_f, vc, rot_mat)
        _add_camera(scene)
        panels.append(_render_scene(scene, img_size, img_size))

        # src panels
        for si, (src_v_seq, src_f) in enumerate(zip(src_vs, src_fs)):
            V_src = _normalize_mesh(src_v_seq[frame_idx].astype(float))
            diff = _compute_face_diff(src_v_seq[frame_idx], src_f, anc_vs[frame_idx])
            if threshold is not None:
                diff = np.clip(diff, 0, threshold)
            diff_norm = np.clip(diff / (diff_max + 1e-12), 0, 1)
            vc = _vc_shade_with_diff(V_src, src_f, rot_mat, light_dir_np, diff_norm, c_map)
            scene = _build_scene(bg_black)
            _add_mesh_node(scene, V_src, src_f, vc, rot_mat)
            _add_camera(scene)
            panels.append(_render_scene(scene, img_size, img_size))

        frames.append(np.concatenate(panels, axis=1))

    imageio.mimwrite(tmp_path, frames, fps=fps, codec='libx264', quality=8)

    if audio_fn is not None:
        _mux_audio(audio_fn, tmp_path, anim_name)
        print(f"saved as: {anim_name}")
    else:
        print(f"saved as: {anim_name}")

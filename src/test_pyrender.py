"""
Headless rendering test for pyrender.py using stanford_bunny.obj.
Outputs saved to test_pyout/.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import trimesh
import numpy as np

# Import the pyrender-based module
import pyrender as _pr_lib  # noqa: make sure pyrender is importable first
import importlib, types

# We need to import our module without name collision with the pyrender package.
# Our file is src/pyrender.py; pyrender package is also named 'pyrender'.
# Load our module explicitly by file path.
import importlib.util
spec = importlib.util.spec_from_file_location("pr_render", os.path.join(os.path.dirname(__file__), "pyrender.py"))
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

mesh = trimesh.load('stanford_bunny.obj', force='mesh')
V = np.array(mesh.vertices)
F = np.array(mesh.faces)
print(f"Loaded: V={V.shape}, F={F.shape}")

os.makedirs('test_pyout', exist_ok=True)

# ── plot_mesh_image ────────────────────────────────────────────────────────────
print("\n[1] plot_mesh_image (mesh / shade / normal)...")
for mode in ['mesh', 'shade', 'normal']:
    pr.plot_mesh_image(
        [V, V], [F, F],
        rot_list=[[0, 30, 0], [0, -30, 0]],
        mode=mode,
        bg_black=(mode == 'normal'),
        save=True,
        savedir='test_pyout',
        name=f'bunny_{mode}',
    )
    print(f"  -> test_pyout/bunny_{mode}.png")

# ── plot_mesh_gouraud (shade) ──────────────────────────────────────────────────
print("\n[2] plot_mesh_gouraud (shade)...")
pr.plot_mesh_gouraud(
    [V, V], [F, F],
    rot_list=[[0, 30, 0], [0, -30, 0]],
    mode='shade',
    bg_black=True,
    save=True,
    show=False,
    logdir='test_pyout',
    name='bunny_gouraud_shade',
)
print("  -> test_pyout/bunny_gouraud_shade.png")

# ── plot_mesh_gouraud (normal) ─────────────────────────────────────────────────
print("\n[3] plot_mesh_gouraud (normal)...")
pr.plot_mesh_gouraud(
    [V], [F],
    rot_list=[[0, 30, 0]],
    mode='normal',
    bg_black=True,
    save=True,
    show=False,
    logdir='test_pyout',
    name='bunny_gouraud_normal',
)
print("  -> test_pyout/bunny_gouraud_normal.png")

# ── plot_image_array_diff3 ─────────────────────────────────────────────────────
print("\n[4] plot_image_array_diff3...")
np.random.seed(0)
V_noisy = V + np.random.randn(*V.shape) * 0.001
pr.plot_image_array_diff3(
    [V_noisy, V_noisy * 1.002], [F, F],
    D=V,
    rot=(0, 30, 0),
    bg_black=False,
    save=True,
    logdir='test_pyout',
    name='bunny_diff3',
)
print("  -> test_pyout/bunny_diff3.png")

# ── plot_mesh_video (short sequence) ──────────────────────────────────────────
print("\n[5] plot_mesh_video (20 frames rotation)...")
num_frames = 20
rot_list = [[0, i * 18, 0] for i in range(num_frames)]
V_seq = np.stack([V] * num_frames)         # (T, V, 3)
pr.plot_mesh_video(
    [V_seq],
    [F],
    rot_list=rot_list,
    size=4,
    bg_black=True,
    fps=10,
    savedir='test_pyout',
    savename='bunny_rotate',
)
print("  -> test_pyout/bunny_rotate.mp4")

# ── render_video_mesh_diff ────────────────────────────────────────────────────
print("\n[6] render_video_mesh_diff (20 frames)...")
V_seq2 = V_seq + np.random.randn(*V_seq.shape) * 0.001
pr.render_video_mesh_diff(
    [V_seq, V_seq2],
    [F],
    D=V_seq,
    rot_list=rot_list,
    size=4,
    bg_black=False,
    fps=10,
    savedir='test_pyout',
    savename='bunny_diff_rotate',
)
print("  -> test_pyout/bunny_diff_rotate.mp4")

# ── render_video_mesh_diffs ───────────────────────────────────────────────────
print("\n[7] render_video_mesh_diffs (20 frames)...")
pr.render_video_mesh_diffs(
    src_vs=[V_seq2],
    src_fs=[F],
    anc_vs=V_seq,
    anc_fs=F,
    rot_list=rot_list,
    size=4,
    bg_black=False,
    fps=10,
    savedir='test_pyout',
    savename='bunny_diffs',
)
print("  -> test_pyout/bunny_diffs.mp4")

print("\n=== All tests passed ===")

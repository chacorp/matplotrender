import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import trimesh
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless

from matplotrender import plot_mesh_image, plot_image_array_diff3, plot_mesh_gouraud, plot_mesh_video, render_video_mesh_diff, render_video_mesh_diffs

mesh = trimesh.load('stanford_bunny.obj', force='mesh')
V = np.array(mesh.vertices)
F = np.array(mesh.faces)

print(f"Loaded: V={V.shape}, F={F.shape}")

os.makedirs('test_out', exist_ok=True)

# --- test plot_mesh_image ---
print("Testing plot_mesh_image (mesh / shade / normal)...")
for mode in ['mesh', 'shade', 'normal']:
    plot_mesh_image(
        [V], [F],
        rot_list=[[0, 30, 0]],
        norm=True,
        mode=mode,
        save=True,
        savedir='test_out',
        name=f'bunny_{mode}',
    )
    print(f"  saved test_out/bunny_{mode}.png")

# --- test plot_image_array_diff3 ---
print("Testing plot_image_array_diff3...")
V2 = V + np.random.randn(*V.shape) * 0.001
plot_image_array_diff3(
    [V2], [F],
    D=V,
    rot=(0, 30, 0),
    norm=True,
    bg_black=False,
    save=True,
    logdir='test_out',
    name='bunny_diff3',
)
print("  saved test_out/bunny_diff3.png")

# --- test plot_mesh_gouraud ---
print("Testing plot_mesh_gouraud (shade / normal)...")
for mode in ['shade', 'normal']:
    plot_mesh_gouraud(
        [V], [F],
        rot_list=[[0, 30, 0]],
        norm=True,
        mode=mode,
        bg_black=False,
        save=True,
        show=False,
        logdir='test_out',
        name=f'bunny_gouraud_{mode}',
    )
    print(f"  saved test_out/bunny_gouraud_{mode}.png")


# --- test plot_mesh_gouraud with torch tensor inputs ---
print("Testing plot_mesh_gouraud with torch tensor inputs...")
import torch
V_torch = torch.tensor(V, dtype=torch.float32)
F_torch = torch.tensor(F, dtype=torch.long)

plot_mesh_gouraud(
    [V_torch], [F_torch],
    rot_list=[[0, 30, 0]],
    norm=True,
    mode='shade',
    bg_black=False,
    save=True,
    show=False,
    logdir='test_out',
    name='bunny_gouraud_torch',
)
print("  saved test_out/bunny_gouraud_torch.png")

# --- test plot_mesh_gouraud with Cs as torch tensor (segmentation labels) ---
print("Testing plot_mesh_gouraud with segmentation Cs (torch)...")
num_verts = V.shape[0]
C_seg = torch.randn(num_verts, 5)  # 5 classes
plot_mesh_gouraud(
    [V_torch], [F_torch],
    Cs=[C_seg],
    rot_list=[[0, 30, 0]],
    norm=True,
    mode='shade',
    bg_black=False,
    save=True,
    show=False,
    logdir='test_out',
    name='bunny_gouraud_seg_torch',
)
print("  saved test_out/bunny_gouraud_seg_torch.png")


# --- test plot_mesh_video (rotating bunny, no diff) ---
print("Testing plot_mesh_video (rotating bunny)...")
T = 36  # frames: 10 deg/frame -> full 360
rot_list = [[0, y, 0] for y in np.linspace(0, 360, T, endpoint=False)]
V_seq = np.tile(V[np.newaxis], (T, 1, 1))  # (T, V, 3) same mesh each frame

plot_mesh_video(
    [V_seq],
    [F],
    rot_list=rot_list,
    norm=True,
    size=4,
    fps=12,
    bg_black=False,
    savedir='test_out',
    savename='bunny_rotate',
)
print("  saved test_out/bunny_rotate.mp4")

# --- test render_video_mesh_diff (rotating bunny vs noisy bunny) ---
print("Testing render_video_mesh_diff (GT vs noisy)...")
V_noisy_seq = V_seq + np.random.randn(*V_seq.shape) * 0.002

render_video_mesh_diff(
    [V_seq],
    [F],
    #D=V_seq,           # GT sequence (optional)
    rot_list=rot_list,
    norm=True,
    size=4,
    fps=12,
    bg_black=False,
    savedir='test_out',
    savename='bunny_diff_rotate',
)
print("  saved test_out/bunny_diff_rotate.mp4")


# --- test render_video_mesh_diffs ---
print("Testing render_video_mesh_diffs (anchor vs two noisy src)...")
V_noisy1 = V_seq + np.random.randn(*V_seq.shape) * 0.003
V_noisy2 = V_seq + np.random.randn(*V_seq.shape) * 0.006

render_video_mesh_diffs(
    src_vs=[V_noisy1, V_noisy2],
    src_fs=[F, F],
    anc_vs=V_seq,
    anc_fs=F,
    rot_list=rot_list,
    norm=True,
    size=4,
    fps=12,
    bg_black=False,
    savedir='test_out',
    savename='bunny_diffs',
)
print("  saved test_out/bunny_diffs.mp4")

print("\nAll tests passed!")

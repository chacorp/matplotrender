import os
from glob import glob
import trimesh
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
from functools import partial
from tqdm import tqdm

from typing import Callable, List, Optional, Union

import subprocess

"""
Reference: https://matplotlib.org/matplotblog/posts/custom-3d-engine/
"""


def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M

def ortho(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 3] = 1.0
    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(zfar + znear) / (zfar - znear)
    return M

def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=float)

def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c, 0, s, 0],
                      [ 0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def zrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c,-s, 0, 0],
                      [ s, c, 0, 0],
                      [ 0, 0, 1, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ 1, 0, 0, 0],
                      [ 0, c,-s, 0],
                      [ 0, s, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def transform_vertices(frame_v, MVP, F, norm=True, no_parsing=False):
    V = frame_v
    if norm:
        V = (V - (V.max(0) + V.min(0)) *0.5) / max(V.max(0) - V.min(0))
    V = np.c_[V, np.ones(len(V))]
    V = V @ MVP.T
    V /= V[:, 3].reshape(-1, 1)
    if no_parsing:
        return V
    VF = V[F]
    return VF

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
    
    if mode=='faces':
        return norm
    
    # Compute mean vertex normals manually
    vertex_normals = np.zeros(vertices.shape, dtype=np.float64)
    for i, face in enumerate(faces):
        for vertex in face:
            vertex_normals[vertex] += norm[i]

    # Normalize the vertex normals
    norm_v = vertex_normals / (np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis] + 1e-12)
    return norm_v
    
def plot_image_array(
        Vs,
        Fs, 
        rot_list=None, 
        size=6, 
        norm=False, 
        view_mode='p',
        mode='mesh', 
        linewidth=1, 
        linestyle='solid', 
        light_dir=np.array([0,0,1]),
        bg_black = True,
        logdir='.', 
        name='000', 
        save=False
        ):
    r"""
    Args:
        Vs (list): list of vertices [V, V, V, ...]
        Fs (list): list of face indices [F, F, F, ...]
        rot_list (list): list of euler angle [ [x,y,z], [x,y,z], ...]
        size (int): size of figure
        norm (bool): if True, normalize vertices
        view_mode (str): if 'p' use perspective, if 'o' use orthogonal camera
        mode (str): mode for rendering [mesh(wireframe), shade, normal]
        linewidth (float): line width for wireframe (kwargs for matplotlib)
        linestyle (str): line style for wireframe (kwargs for matplotlib)
        light_dir (np.array): light direction
        bg_black (bool): if True, use dark_background for plt.style
        logdir (str): directory for saved image
        name (str): name for saved image
        save (bool): if True, save the plot as image
    """
    if mode=='gouraud':
        print("currently WIP!: need to curl by z")
        
    num_meshes = len(Vs)
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig = plt.figure(figsize=(size * num_meshes, size))  # Adjust figure size based on the number of meshes
    
    for idx, (V, F) in enumerate(zip(Vs, Fs)):
        # Calculate the position of the subplot for the current mesh
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)

        #xrot, yrot, zrot = rot[0], 90, rot[2]
        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        model = translate(0, 0, -4) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        if view_mode=='p':
            proj  = perspective(55, 1, 1, 100)
        else:
            proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
            
        MVP   = proj @ model # view is identity
        
        # quad to triangle    
        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        if mode=='normal':
            C = calc_face_norm(V[F]) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            C = np.clip(C, 0, 1) if False else C * 0.5 + 0.5
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        elif mode=='shade':
            C = calc_face_norm(V[F]) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            
            C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)
            C = C*0.5+0.25
            collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        elif mode=='gouraud':
            # I = np.argsort(Z)
            # V, F, vidx = get_new_mesh(V, F, I, invert=True)
            
            ### curling by normal
            C = calc_norm(V, F, mode='v') #@ model[:3,:3].T
            NI = np.argwhere(C[:,2] > 0.0).squeeze()
            V, F, vidx = get_new_mesh(V, F, NI, invert=True)
            
            C = calc_norm(V, F,mode='v') #@ model[:3,:3].T
            
            #VV = (V-V.min()) / (V.max()-V.min())# world coordinate
            V = transform_vertices(V, MVP, F, norm, no_parsing=True)
            triangle_ = tri.Triangulation(V[:,0], V[:,1], triangles=F)
            
            C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)
            C = C*0.5+0.25
            #VV = (V-V.min()) / (V.max()-V.min()) #screen coordinate
            #cmap = colors_to_cmap(VV)
            cmap = colors_to_cmap(C)
            zs = np.linspace(0.0, 1.0, num=V.shape[0])
            plt.tripcolor(triangle_, zs, cmap=cmap, shading='gouraud')
            
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor='black')
            
        if mode!='gouraud':
            ax.add_collection(collection)
        plt.xticks([])
        plt.yticks([])
    
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def setup_plot(bg_black, size, num_meshes):
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig, axes = plt.subplots(1, num_meshes, figsize=(size * num_meshes, size))
    if num_meshes == 1:
        axes = [axes]
    for ax in axes:
        ax.set_xlim(-1, 1)  # Adjusted to prevent cutting off the mesh
        ax.set_ylim(-1, 1)  # Adjusted to prevent cutting off the mesh
        ax.set_aspect('equal')
        ax.axis('off')
    return fig, axes

def transform_and_project(V, F, MVP, norm):
    VF_tri = transform_vertices(V, MVP, F, norm)
    T = VF_tri[:, :, :2]
    Z = -VF_tri[:, :, 2].mean(axis=1)
    zmin, zmax = Z.min(), Z.max()
    Z = (Z - zmin) / (zmax - zmin)
    return T, Z

def prepare_color(C, model, light_dir):
    C = C @ model[:3, :3].T
    C = (C @ light_dir)[:, np.newaxis].repeat(3, axis=-1)
    C = np.clip(C, 0, 1)
    #C = C * 0.6 + 0.3
    return C

def process_mesh(V, F, MVP, norm, model, light_dir, linewidth, c_map, diff=None):
    T, Z = transform_and_project(V, F, MVP, norm)
    C = calc_face_norm(V, F)
    C = prepare_color(C, model, light_dir)
    I = np.argsort(Z)
    T, C = T[I, :], C[I, :]
    C = np.clip(C, 0, 1)
    
    NI = np.argwhere(C[:, 2] > 0).squeeze()
    T, C = T[NI, :], C[NI, :]
    if diff is not None:
        diff = diff[I]
        diff = diff[NI]
        Dc = plt.get_cmap(c_map)(diff)
        mask = diff[:, np.newaxis]
        C = C * (1 - mask) + Dc[:, :3] * mask
        C = np.clip(C, 0, 1)
    C = C * 0.7 + 0.2
    return T, C

def update_frame_diff(frame_idx, Vs, Fs, D, axes, linewidth, c_map, norm, light_dir, threshold, rot_list):
    for ax in axes:
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    V = D[frame_idx]
    F = Fs[0]
    
    xrot, yrot, zrot = rot_list[frame_idx] if rot_list is not None else (0, 0, 0)
    
    model = translate(0, 0, -5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    # proj  = perspective(65, 1, 1, 10)
    proj = ortho(-1, 1, -1, 1, 1, 100)
    MVP = proj @ model
    
    # Plot the GT mesh
    T, C = process_mesh(V, F, MVP, norm, model, light_dir, linewidth, c_map)
    collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
    axes[0].add_collection(collection)
    axes[0].set_title(f'Frame {frame_idx + 1:03d} * 1e-2', fontsize=6)
    axes[0].axis('off')
    
    #D_diff = np.array([abs(np.array(D[frame_idx]) - np.array(vs[frame_idx])) for vs in Vs])[:, F]
    D_diff = np.array(abs(np.array(D[frame_idx]) - np.array(Vs[:, frame_idx])))[:, F]
    D_diff = np.linalg.norm(D_diff, axis=-1)
    D_diff = np.linalg.norm(D_diff, axis=-1)
    
    if threshold is not None:
        D_diff[D_diff > threshold] = 0
    diff_min, diff_max = D_diff.min(), D_diff.max()
    
    for idx, V in enumerate(Vs):
        diff = D_diff[idx]
        if diff_max > 0:
            diff = (diff - diff_min) / (diff_max - diff_min)
        
        T, C = process_mesh(V[frame_idx], F, MVP, norm, model, light_dir, linewidth, c_map, diff)
        collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        axes[idx + 1].add_collection(collection)
#         axes[idx + 1].set_xlabel(f'Frame {frame_idx + 1} | min:{D_diff.min():.5f} | max: {D_diff.max():.5f}')
        axes[idx + 1].set_title(f'min:{D_diff[idx].min()*100:.3f} | max: {D_diff[idx].max()*100:.3f}', fontsize=6)
        axes[idx + 1].axis('off')
        #plt.xlabel(f'Frame {frame_idx + 1} | min:{D_diff.min():.5f} | max: {D_diff.max():.5f}')
    # Add the text to the figure
    #fig.text(0.5, 0.01, f'min: {diff_min:.4f} | max: {diff_max:.4f}', ha='center', fontsize=12, transform=fig.transFigure)

def render_mesh_diff(
        Vs, 
        Fs, 
        D, 
        rot_list=None,
        size:int=6,
        norm:bool=False,
        linewidth:float=1,
        light_dir=np.array([0,0,1]),
        bg_black:bool=True,
        threshold:float=None,
        c_map:str='YlOrRd', 
        fps:int=30,
        savedir:Optional[str]=None, 
        savename:Optional[str]="temp",
        audio_dir:Optional[str]=None,
    ):
    """
    >>>
    v_list=[ pred_outputs[:frame_num], vertices.numpy()[:frame_num] ]
    f_list=[ ict_full.faces ]
    d_list=[ vertices.numpy()[:frame_num] ]
    render_mesh_diff(
        v_list, 
        f_list, 
        d_list[0],
        size=2, bg_black=False,
        savedir='_tmp',
        savename="temp",
    )
    """
    num_frames = len(D)
    num_meshes = len(Vs) + 1
    fig, axes = setup_plot(bg_black, size, num_meshes)
    Vs = np.array(Vs)
    
    plt.tight_layout()
    #fig.subplots_adjust(top=0.95)  # Adjust the top margin to ensure titles are not cut off
    anim = FuncAnimation(
        fig, 
        update_frame_diff, 
        frames=num_frames, 
        fargs=(Vs, Fs, D, axes, linewidth, c_map, norm, light_dir, threshold, rot_list),
        repeat=False
    )
    
    if savedir is None:
        plt.show()
    else:
        if audio_dir is None:
            anim_name = f'{savedir}/{savename}.mp4'
        else:
            anim_name = f'{savedir}/_tmp_.mp4'
        bar = tqdm(total=num_frames, desc="rendering")
        anim.save(
            anim_name, 
            fps=fps,
            progress_callback=lambda i, n: bar.update(1)
        )
            
    plt.close()
    
    if audio_dir is not None:
        # mux audio and video
        print("[INFO] mux audio and video")
        cmd = f"ffmpeg -y -i {audio_dir} -i {savedir}/_tmp_.mp4 -c:v copy -c:a aac {savedir}/{savename}.mp4"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"saved as: {savedir}/{savename}.mp4")

        # remove tmp files
        subprocess.call(f"rm -f {savedir}/_tmp_.mp4", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
def compute_average_distance(points):
    """
    Compute the average distance of each point from the origin in a set of points.
    Args:
    ----
        points: np.ndarray
            A set of points.
    Returns:
    -------
        distances: np.ndarray
            Average distance.
    """
    distances = np.linalg.norm(points, axis=1)
    distances = np.mean(distances)
    return distances

def procrustes_analysis(P, Q):
    """
    Extended procrustes analysis to find the optimal rotation, translation, and scaling
    that aligns two sets of points P and Q minimizing the RMSD.
    
    Args
    ----
        p : np.ndarray
            A set of points.
        Q : np.ndarray
            A set of corresponding points.
    Returns
    -------
        R : np.ndarray
            Rotation matrix
        t : np.ndarray
            translation
        s : np.ndarray
            scale factor
    """

    # Calculate the centroids of the point sets
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Center the points around the origin
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Calculate the average distances from the origin
    avg_dist_P = compute_average_distance(P_centered)
    avg_dist_Q = compute_average_distance(Q_centered)

    # Calculate the scale factor
    s = avg_dist_Q / avg_dist_P

    # Scale the points
    P_scaled = P_centered * s

    # Compute the covariance matrix
    H = P_scaled.T @ Q_centered

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = Vt.T @ U.T

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute the translation vector
    t = -R @ (centroid_P * s) + centroid_Q

    return R, t, s
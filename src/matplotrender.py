import os
from glob import glob
import trimesh
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as matclrs
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
from functools import partial
from tqdm import tqdm

from typing import Callable, List, Optional, Union

import subprocess

from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.cm import get_cmap

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

def scale(x, y, z):
    return np.array([[x, 0, 0, 1],
                     [0, y, 0, 1],
                     [0, 0, z, 1],
                     [0, 0, 0, 1]], dtype=float)
    
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
    matrix = yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    return matrix

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

def colors_to_cmap(colors):
    '''
    colors_to_cmap(nx3_or_nx4_rgba_array) yields a matplotlib colormap object that, when
    that will reproduce the colors in the given array when passed a list of n evenly
    spaced numbers between 0 and 1 (inclusive), where n is the length of the argument.

    Example:
      cmap = colors_to_cmap(colors)
      zs = np.asarray(range(len(colors)), dtype=float) / (len(colors)-1)
      # cmap(zs) should reproduce colors; cmap[zs[i]] == colors[i]
    '''
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        colors = np.hstack((colors, np.ones((len(colors),1))))
    steps = (0.5 + np.asarray(range(len(colors)-1), dtype=float))/(len(colors) - 1)
    return matclrs.LinearSegmentedColormap(
        'auto_cmap',
        {clrname: ([(0, col[0], col[0])] + 
                   [(step, c0, c1) for (step,c0,c1) in zip(steps, col[:-1], col[1:])] + 
                   [(1, col[-1], col[-1])])
         for (clridx,clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
         for col in [colors[:,clridx]]},
        N=len(colors)
    )

def get_new_mesh(vertices, faces, v_idx, invert=False):
    """Calculate standardized mesh
    Args
    ------
        vertices (np.ndarray): [V, 3] array of vertices 
        faces (np.ndarray): [F, 3] array of face indices 
        v_idx (np.ndarray): [N] list of vertex index to remove from mesh
    Return
    ------
        updated_verts (np.ndarray): [V, 3] new array of vertices 
        updated_faces (np.ndarray): [F, 3] new array of face indices  
        updated_verts_idx (np.ndarray): [N] list of vertex index to remove from mesh (fixed)
    """
    max_index = vertices.shape[0]
    new_vertex_indices = np.arange(max_index)

    if invert:
        mask = np.zeros(max_index, dtype=bool)
        mask[v_idx] = True
    else:
        mask = np.ones(max_index, dtype=bool)
        mask[v_idx] = False

    updated_verts = vertices[mask]
    updated_verts_idx = new_vertex_indices[mask]

    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(updated_verts_idx)}

    updated_faces = np.array([
                    [index_mapping.get(idx, -1) for idx in face]
                    for face in faces
                ])

    valid_faces = ~np.any(updated_faces == -1, axis=1)
    updated_faces = updated_faces[valid_faces]
    
    return updated_verts, updated_faces, updated_verts_idx

def plot_mesh_image(
        Vs, 
        Fs, 
        rot_list=None, 
        size=6, 
        norm=False, 
        mode='mesh',
        z_pos=-5, 
        custom_perspective=None,
        linewidth=1, 
        linestyle='solid', 
        light_dir=np.array([0,0,1]),
        bg_black=False,
        savedir='.', 
        name='000', 
        save=False,
    ):
    """
    Args:
        Vs (list): list of vertices [V, V, V, ...]
        Fs (list): list of face indices [F, F, F, ...]
        rot_list (list): list of euler angle [ [x,y,z], [x,y,z], ...]
        size (int): size of figure
        norm (bool): if True, normalize vertices
        mode (str): mode for rendering [mesh(wireframe), shade, normal]
        z_pos (float): z distance of the mesh 
        custom_perspective (ortho() / perspective() ): if specified, use it to render perspective
        linewidth (float): line width for wireframe (kwargs for matplotlib)
        linestyle (str): line style for wireframe (kwargs for matplotlib)
        light_dir (np.array): light direction
        bg_black (bool): if True, use dark_background for plt.style
        savedir (str): directory for saved image
        name (str): name for saved image
        save (bool): if True, save the plot as image

    >>>
    v_list = [ mesh1.v ]
    f_list = [ mesh1.f ]
    rot_list = [[0,60,0]] # rotates 60 degree on y axis

    plot_mesh_image(v_list, f_list, rot_list)
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

        if rot_list:
            rot_mat = get_rotation_matrix(rot_list[idx])
        else:
            rot_mat = get_rotation_matrix([0, 0, 0])

        ## MVP
        model = translate(0, 0, z_pos) @ rot_mat
        if custom_perspective is None:    
            # proj = perspective(55, 1, 1, 100)
            proj = ortho(-1, 1, -1, 1, 1, 100)
        else:
            proj = custom_perspective
        MVP = proj @ model # view is identity
        
        # quad to triangle    
        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        if mode=='normal':
            C = calc_face_norm(V, F) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            C = np.clip(C, 0, 1) if False else C * 0.5 + 0.5
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        elif mode=='shade':
            C = calc_face_norm(V, F) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] >= 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            
            C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)
            C = C*0.7+0.2
            collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        elif mode=='gouraud':
            # I = np.argsort(Z)
            # V, F, vidx = get_new_mesh(V, F, I, invert=True)
            
            ### curling by normal
            C = calc_face_norm(V, F, mode='v') #@ model[:3,:3].T
            NI = np.argwhere(C[:,2] > 0.0).squeeze()
            V, F, vidx = get_new_mesh(V, F, NI, invert=True)
            
            C = calc_face_norm(V, F,mode='v') #@ model[:3,:3].T
            
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
        plt.savefig('{}/{}.png'.format(savedir, name), bbox_inches = 'tight')
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
    
    if rot_list:
        rot_mat = get_rotation_matrix(rot_list[frame_idx])
    else:
        rot_mat = get_rotation_matrix([0, 0, 0])
    
    model = translate(0, 0, -5) @ rot_mat
    # proj  = perspective(65, 1, 1, 10)
    proj = ortho(-1, 1, -1, 1, 1, 100)
    MVP = proj @ model
    
    # Plot the GT mesh
    T, C = process_mesh(V, F, MVP, norm, model, light_dir, linewidth, c_map)
    collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
    axes[0].add_collection(collection)
    axes[0].set_title(f'Frame {frame_idx + 1:03d} * 1e-2', fontsize=6)
    axes[0].axis('off')
    
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
        # axes[idx + 1].set_xlabel(f'Frame {frame_idx + 1} | min:{D_diff.min():.5f} | max: {D_diff.max():.5f}')
        axes[idx + 1].set_title(f'min:{D_diff[idx].min()*100:.3f} | max: {D_diff[idx].max()*100:.3f}', fontsize=6)
        axes[idx + 1].axis('off')
        #plt.xlabel(f'Frame {frame_idx + 1} | min:{D_diff.min():.5f} | max: {D_diff.max():.5f}')
    # Add the text to the figure
    #fig.text(0.5, 0.01, f'min: {diff_min:.4f} | max: {diff_max:.4f}', ha='center', fontsize=12, transform=fig.transFigure)

def update_frame(frame_idx, Vs, Fs, D, axes, linewidth, c_map, norm, light_dir, threshold, rot_list):
    for ax in axes:
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    F = Fs[0]
    
    if rot_list:
        rot_mat = get_rotation_matrix(rot_list[frame_idx])
    else:
        rot_mat = get_rotation_matrix([0, 0, 0])
    
    model = translate(0, 0, -5) @ rot_mat
    proj = ortho(-1, 1, -1, 1, 1, 100)
    MVP = proj @ model
    
    for idx, V in enumerate(Vs):
        T, C = process_mesh(V[frame_idx], F, MVP, norm, model, light_dir, linewidth, c_map)
        collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        axes[idx].add_collection(collection)
        axes[idx].set_title(f'Frame {frame_idx + 1:03d} * 1e-2', fontsize=6)
        axes[idx].axis('off')

def plot_mesh_video(
        Vs, 
        Fs, 
        D=None,
        rot_list=None,
        size:int=6,
        norm:bool=False,
        linewidth:float=1,
        light_dir=np.array([0,0,1]),
        bg_black:bool=False,
        threshold:float=None,
        c_map:str='YlOrRd', 
        fps:int=30,
        savedir:Optional[str]='tmp', 
        savename:Optional[str]="test",
        audio_dir:Optional[str]=None,
        debug=False,
    ):
    """
    render a video with a case of:
        vert_sequence_a: (T, V, 3)
        vert_sequence_b: (T, V, 3)
        faces: (F, 3)
        vertices: (V, 3)
    >>> 
    v_list=[ vert_sequence_a, vert_sequence_b ]
    f_list=[ faces ]
    d_list=[ vertices ]
    plot_mesh_video(
        v_list, 
        f_list, 
        d_list[0],
        size=2,
    )
    """
    num_frames = len(Vs[0])
    num_meshes = len(Vs) if D is None else len(Vs) + 1
    fig, axes = setup_plot(bg_black, size, num_meshes)
    Vs = np.array(Vs)
    # if debug:
    #     print(len(Vs))
    
    plt.tight_layout()
    #fig.subplots_adjust(top=0.95)  # Adjust the top margin to ensure titles are not cut off
    anim = FuncAnimation(
        fig, 
        update_frame_diff if D is not None else update_frame, 
        frames=num_frames, 
        fargs=(Vs, Fs, D, axes, linewidth, c_map, norm, light_dir, threshold, rot_list),
        repeat=False
    )
    
    # save path
    os.makedirs(savedir, exist_ok=True)
    
    anim_name = f'{savedir}/{savename}.mp4'

    bar = tqdm(total=num_frames, desc="rendering")
    anim.save(
        anim_name if audio_dir is None else 'tmp.mp4', 
        fps=fps,
        progress_callback=lambda i, n: bar.update(1)
    )
    plt.close()
    
    if audio_dir is not None:
        # mux audio and video
        print("[INFO] mux audio and video")
        cmd = f"ffmpeg -y -i {audio_dir} -i tmp.mp4 -c:v copy -c:a aac {anim_name}"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # remove tmp files
        subprocess.call(f"rm -f tmp.mp4", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"saved as: {anim_name}")

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

def plot_image_array_diff3(
    Vs, 
    Fs, 
    D, 
    rot=(0, 0, 0), 
    size=6, 
    norm=False, 
    linewidth=1, 
    linestyle='solid', 
    light_dir=np.array([0,0,1]), 
    bg_black=True, 
    threshold=None, 
    logdir='.', 
    name='000', 
    save=False, 
    draw_base=True, 
    c_map='YlOrRd'
    ):
    """
    WIP ...
    v_list = [vertices] # (V, 3)
    f_list = [faces] # (F, 3)
    v0 = vertices # (V, 3)
    plot_image_array_diff3(
        v_list, 
        f_list, 
        v0,
        rot=[0,-10,0],
        bg_black=False,
    )
    """
    num_meshes = len(Vs) + 1
    fig, axes = setup_plot(bg_black, size, num_meshes)
    
    xrot, yrot, zrot = rot if rot is not None else (0, 0, 0)
    
    model = translate(0, 0, -5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    proj = ortho(-1, 1, -1, 1, 1, 100)
    # proj  = perspective(65, 1, 1, 10)
    MVP = proj @ model

    if draw_base:
        T, C = process_mesh(D, Fs[0], MVP, norm, model, light_dir, linewidth, c_map)
        collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        axes[0].add_collection(collection)
        axes[0].axis('off')
        
    D_diff = np.array(abs(D - np.array(Vs)))[:, Fs[0]]
    D_diff = np.linalg.norm(D_diff, axis=-1)
    D_diff = np.linalg.norm(D_diff, axis=-1)
    
    if threshold is not None:
        D_diff[D_diff > threshold] = threshold
    diff_min, diff_max = D_diff.min(), D_diff.max()
    
    norm_color = plt.Normalize(vmin=0, vmax=diff_max)
    sm = plt.cm.ScalarMappable(cmap=c_map, norm=norm_color)
    sm.set_array([])

    for idx, (V, F) in enumerate(zip(Vs, Fs)):
        diff = D_diff[idx]
        if diff_max > 0:
            diff = (diff - diff_min) / (diff_max - diff_min)
        
        T, C = process_mesh(V, F, MVP, norm, model, light_dir, linewidth, c_map, diff)
        collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        axes[idx + 1].add_collection(collection)
        axes[idx + 1].set_title(f'mean: {D_diff[idx].mean():.2e} | std: {D_diff[idx].std():.2e}')
        axes[idx + 1].axis('off')

    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.006, pad=0.015)
    cbar.set_label('Mean Squared Error', rotation=90, labelpad=15)
    
    if save:
        plt.savefig(f'{logdir}/{name}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def render_video_mesh_diff(Vs, Fs, D, 
                    rot_list=None,
                    size=6,
                    norm=False,
                    linewidth=1,
                    light_dir=np.array([0,0,1]),
                    bg_black=True,
                    threshold=None,
                    c_map='YlOrRd', 
                    savedir=None, 
                    savename="temp",
                    audio_fn=None,
                    fps=30
                    ):
    """
    ex):
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
        fig, update_frame, 
        frames=num_frames, 
        fargs=(Vs, Fs, D, axes, linewidth, c_map, norm, light_dir, threshold, rot_list), 
        repeat=False
    )
    
    if savedir is None:
        plt.show()
    else:
        if audio_fn is None:
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
    
    if audio_fn is not None:
        # mux audio and video
        print("[INFO] mux audio and video")
        cmd = f"ffmpeg -y -i {audio_fn} -i {savedir}/_tmp_.mp4 -c:v copy -c:a aac {savedir}/{savename}.mp4"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"saved as: {savedir}/{savename}.mp4")

        # remove tmp files
        subprocess.call(f"rm -f {savedir}/_tmp_.mp4", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(-1)[:,None]

def transform(V, M):
    V_homo = np.hstack([V, np.ones((V.shape[0], 1))])
    return (M @ V_homo.T).T[:, :3]

def normalize(V):
    V = V - V.mean(axis=0)
    return V / np.max(np.linalg.norm(V, axis=1))

def plot_mesh_gouraud(Vs, Fs, Cs=None, rot_list=None, size=6, norm=False, 
                         mode='shade', # not used always gouraud shading applied
                         threshold=0.01,
                         seg_divide=False,
                         is_diff=False,
                         diff_base=None,
                         diff_revert=False,
                         backface_culling=False,
                         depth_sorting=True,
                         linewidth=1, linestyle='solid', 
                         light_dir=np.array([0, 0, 1]),
                         light_at_frontface=True, ## light sticked to the front of the face
                         view_dir=np.array([0, 0, 1]),
                         mesh_scale=1.0,
                         mesh_trans=np.array([0,0,0]),
                         c_map="nipy_spectral", blend=0.5,
                         seg_only=-1,
                         bg_black=True, logdir='.', name='000', save=False, show=True):

    num_meshes = len(Vs)
    
    plt.style.use('dark_background' if bg_black else 'default')
    fig = plt.figure(figsize=(size * num_meshes, size))
    
    if is_diff:
        if diff_base is None:
            raise ValueError('diff_base is None!')
            
    if Cs==None:
        Cs = [None] * num_meshes
    
    # light source data type int -> float
    light_dir_view = light_dir.astype(float)
            
    for idx, (V, F, C) in enumerate(zip(Vs, Fs, Cs)):
                
        if norm:
            V = normalize(V)
            
        if is_diff:
            C = np.linalg.norm(np.square(V-diff_base), axis=-1)
            if C.max() > 0:
                C = (C - C.min(0)) / (C.max(0) - C.min(0))
            C = 1 - C if diff_revert else C
            print(C.shape, C.max(0), C.min(0))
            
        # scale and translate for render
        V = V * mesh_scale + mesh_trans
    
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, 1], ylim=[-1, 1], aspect=1, frameon=False)

        xrot, yrot, zrot = rot_list[idx] if rot_list else (0, 0, 0)
        # very naiiiiiive rotation stacks 
        model = translate(0, 0, -5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        proj = ortho(-1, 1, -1, 1, 1, 100)
        MVP = proj @ model

        model_rot = model[:3, :3]
        model_rot_invT = np.linalg.inv(model_rot).T

        # 정점 normal
        vertex_normals_obj = calc_face_norm(V, F, mode='v')
        vertex_normals_view = (model_rot_invT @ vertex_normals_obj.T).T
        
        if backface_culling:
            face_normals_obj = calc_face_norm(V, F)
            face_normals_view = (model_rot_invT @ face_normals_obj.T).T
            keep = (face_normals_view @ view_dir) >= 0
            F = F[keep]
        
        # light sticked to front face
        if light_at_frontface:
            shading = np.clip(vertex_normals_obj @ light_dir_view, 0, 1)
        else:
            shading = np.clip(vertex_normals_view @ light_dir_view, 0, 1)
        
        ## projection!
        V_proj = transform(V, MVP)

        # depth sorting: triangle 중심 z
        if depth_sorting:
            tri_depth = V_proj[F][:, :, 2].mean(axis=1)
            sort_idx = np.argsort(-tri_depth)
            F = F[sort_idx]

        # make triangle
        triang = tri.Triangulation(V_proj[:, 0], V_proj[:, 1], F)
        
        vertex_color = shading[..., np.newaxis].repeat(3, axis=-1)
        vertex_color = vertex_color *0.6 + 0.3
        
        if is_diff:
            Sc = plt.get_cmap("YlOrRd")(C)[...,:3] ## [N, 4]
            mask = C[:,np.newaxis]
            vertex_color = vertex_color*(1-mask) + Sc*mask
            
        else:
            if C != None:
                len_seg = C.shape[-1]
                S = softmax(C) # softmax
                S = S.argmax(-1)#.numpy()#.float()

                # only render triangle that matches the segment label!
                if seg_only > 0:
                    SF = S[F]

                    v_mask = (S == seg_only).any(-1)
                    vertex_color = vertex_color[v_mask][0]
                    S = S[v_mask][0]

                    f_mask = (SF == seg_only).any(-1)
                    F_sorted_masked = F[f_mask]
                    triang = tri.Triangulation(V_proj[:, 0], V_proj[:, 1], F_sorted_masked)

                S = S.astype(float)
                Sc = plt.get_cmap(c_map)(S/len_seg)[...,:3]

                vertex_color = vertex_color*(1-blend) + blend*Sc
                vertex_color = np.clip(vertex_color, 0, 1)
        
        cmap = colors_to_cmap(vertex_color)
        zs = np.linspace(0.0, 1.0, num=V.shape[0])
        
        # Gouraud shading
        plt.tripcolor(triang, zs, cmap=cmap, shading='gouraud')
        
        ax.set_xticks([])
        ax.set_yticks([])

    if save:
        plt.savefig(f'{logdir}/{name}.png', bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()

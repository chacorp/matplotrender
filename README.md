# Matplotlib render
render 3D mesh with matplotlib library \
I made this just for fun, so the code is a bit messy and there are things that still needs to be done too!


Reference: https://matplotlib.org/matplotblog/posts/custom-3d-engine/


## TODOs
- Image
    - [x] curling based on normal
    - [x] rendering mesh difference (e.g. MSE ...)
    - [x] gouraud rendering with `matplotlib.tri`
- Animation
    - [ ] add slider feature to tour around the animation frames
    - [ ] add feature for comparing difference between two animation
    - [ ] interactive ui

# Requirements
```
pip install -r requirements.txt
pip install .
```

# NEW!
render with gouraud shading! \
also supports visualizing difference/error
```python
from matplotrender import *
import trimesh

mesh0 = trimesh.load('your_mesh_file0.obj')
mesh1 = trimesh.load('your_mesh_file1.obj')

v_list = [mesh1.vertices, mesh2.vertices]
f_list = [mesh1.faces, mesh2.faces]
rot_list=[[0,0,0] ,  [0,0,0]] # you can control rotation for individual mesh

plot_mesh_gouraud(
    v_list, 
    f_list, 
    is_diff=True, 
    diff_base=mesh2.vertices, # calculates difference based on this mesh
    rot_list=rot_list,
)
```
The meshes will be rendered from left to right (mesh1, mesh2)
<img src="demo3.png" />

# How to use
```python
from matplotrender import *
import trimesh
mesh = trimesh.load('your_mesh_file.obj')
print(mesh.vertices.shape)
>>> (5023, 3)

# figure size
SIZE = 2

# mesh that you wanna render
v_list=[ mesh.vertices ]
f_list=[ mesh.faces ]

# xyz Euler angle to rotate the mesh
rot_list=[ [0,0,0] ]

plot_mesh_image(v_list, f_list, rot_list=rot_list, size=SIZE, mode='mesh') # default
plot_mesh_image(v_list, f_list, rot_list=rot_list, size=SIZE, mode='normal')
plot_mesh_image(v_list, f_list, rot_list=rot_list, size=SIZE, mode='shade')
```
<img src="demo.png" width="256" height="740" />

You can also render video! \
let's say you have an mesh animation saved as a npy file as below:
```python
# your mesh animation
vertices_anim = np.load('your_mesh_animation.npy')
print(vertices_anim)
>>> (100, 5023, 3)
```
you can use `plot_mesh_video` to render a video.
```python
v_list=[ vertices_anim ]
f_list=[ mesh.faces ]
plot_mesh_video(
        v_list, 
        f_list, 
        size=2, 
        bg_black=False,
    )
```

you can also mux audio too!
```python
audio_dir='your_path/audio.wav'
v_list=[ vertices_anim ]
f_list=[ mesh.faces ]
plot_mesh_video(
        v_list, 
        f_list, 
        size=2, 
        bg_black=False,
        audio_dir=audio_dir
    )
```



## Rendering difference (errors)
you can render L1 error between meshes (only for same topology)
```python
from matplotrender import *
import trimesh

mesh0 = trimesh.load('your_mesh_file0.obj')
mesh1 = trimesh.load('your_mesh_file1.obj')
mesh2 = trimesh.load('your_mesh_file2.obj')
mesh3 = trimesh.load('your_mesh_file3.obj')

v_list = [mesh1.vertices, mesh2.vertices, mesh3.vertices]
f_list = [mesh1.faces, mesh2.faces, mesh3.faces]
v0 = mesh0.vertices # (V, 3)

plot_image_array_diff3(
    v_list, 
    f_list, 
    v0,
    rot=[0,-10,0], 
    bg_black=False,
)
```
The meshes will be rendered from left to right(mesh0, mesh1, mesh2, mesh3)
<img src="demo2.png" />

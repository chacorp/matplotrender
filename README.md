# Matplotlib render
render 3D mesh with matplotlib library \
I made this just for fun, so the code is a bit messy and there are things that still needs to be done too!


Reference: https://matplotlib.org/matplotblog/posts/custom-3d-engine/


# Requirements
```
pip install -r requirements.txt
pip install .
```


# How to use
```python
from matplotrender import *
import trimesh
mesh = trimesh.load('your_mesh_file.obj')

# figure size
SIZE = 2

# mesh that you wanna render
v_list=[ mesh.vertices ]
f_list=[ mesh.faces ]

# xyz Euler angle to rotate the mesh
rot_list=[ [0,0,0] ]

plot_mesh_image(v_list, f_list, rot_list=rot_list, size=SIZE)
plot_mesh_image(v_list, f_list, rot_list=rot_list, size=SIZE, mode='normal')
plot_mesh_image(v_list, f_list, rot_list=rot_list, size=SIZE, mode='shade')
```
<img src="https://github.com/chacorp/matplotlib_render/blob/main/demo.png" width="256" height="740" />

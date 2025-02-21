import numpy as np
import gmsh
import dolfinx
import ufl
from mpi4py import MPI
from numba import jit
from petsc4py import PETSc
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

comm = MPI.COMM_WORLD

print('Define Parameters')

# ================================================================
# Declare all constants
# ================================================================

# Define structural properties
n_panel = 10   # number of panels

E_panel = 7.0e10  # Value from CSM3, turek-hron, youngs modulus is set suoer high to get rid of elastic deformation
nu_panel = 0.22  # Value from CSM3, turek-hron
mu_panel = E_panel / (2.0 * (1.0 + nu_panel))
lamda_panel = nu_panel*E_panel/(1+nu_panel)/(1-2*nu_panel)  # Value from CSM3, turek-hron

E_block = 7.0e12  # Value from CSM3, turek-hron, youngs modulus is set suoer high to get rid of elastic deformation
nu_block = 0.22  # Value from CSM3, turek-hron
mu_block = E_block / (2.0 * (1.0 + nu_block))
lamda_block = nu_block*E_block/(1+nu_block)/(1-2*nu_block)  # Value from CSM3, turek-hron

r_out = 0.1  # out radius of tube
r_in = 0.09 # inner radius of tube

E_tube = 200.2*10**9*(1-(r_in/r_out)**4)# shear modulus of tortional tube
nu_tube = 0.3
mu_tube = E_tube / (2.0 * (1.0 + nu_tube))
lamda_tube = nu_tube*E_tube/(1+nu_tube)/(1-2*nu_tube)  # Value from CSM3, turek-hron

G = E_tube/2/(1+nu_tube)
Ip = np.pi/2*(r_out**4) # moment of inertia of tube
Iz = np.pi/4*(r_out**4)


panel_length = 2.0   
panel_width = 4.0
panel_thick = 0.02

panel_panel_gap = 0.03   # the gap between panels is 0.03m

total_length = panel_length*n_panel + panel_panel_gap*(n_panel-1)

block_thick = r_out+0.1 # the gap between panel and tube is 0.1m

block_length = 0.02*panel_length
block_width = block_length

stow_angle = np.pi/9 # theta_i

l_0_prime = ((panel_width/2)**2+block_thick**2)**0.5

theta_prime = np.arcsin((panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle))/l_0_prime)

#######################
# generate mesh
#######################

print('Generate the mesh')

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0) 

gmsh.model.add("panel_with_springs")

panel_list = []
block_whole_list = []

for i in range(n_panel):

    panel = gmsh.model.occ.add_box((panel_length+panel_panel_gap)*i, -panel_width/2, block_thick, panel_length, panel_width, panel_thick)
    block_whole = gmsh.model.occ.add_box((panel_length/2-block_length/2)+i*(panel_length+panel_panel_gap), -block_width/2, 0, block_length, block_width, block_thick)
    
    panel_list.append((3, panel))
    block_whole_list.append((3, block_whole))
    
tube_out = gmsh.model.occ.add_cylinder(0, 0, 0, n_panel*panel_length+(n_panel-1)*panel_panel_gap, 0, 0, r_out, angle=2*np.pi)
# tube_in = gmsh.model.occ.add_cylinder(0, 0, 0, n_panel*panel_length+(n_panel-1)*panel_panel_gap, 0, 0, r_in)


block = gmsh.model.occ.cut(block_whole_list, [(3, tube_out)], removeObject=True, removeTool=False)

# tube = gmsh.model.occ.cut([(3, tube_out)], [(3, tube_in)], removeObject=True, removeTool=True)

point_left = gmsh.model.occ.add_point(0,0,0)
point_right = gmsh.model.occ.add_point(n_panel*panel_length+(n_panel-1)*panel_panel_gap, 0, 0)
center_line = gmsh.model.occ.addLine(point_left, point_right)

domain = gmsh.model.occ.fragment(panel_list, block[0]+[(3, tube_out)]+[(1, center_line)]+[(0, point_left)]+[(0, point_right)])


domain = gmsh.model.occ.rotate(domain[0], 0,0,0,1,0,0,stow_angle)


gmsh.model.occ.synchronize()
gmsh.write("panel_tube.brep")


# create physical group

vol_tag_list = gmsh.model.occ.getEntities(3)

for k, vol_tag in enumerate(vol_tag_list):
    vol_id = vol_tag[1]
    gmsh.model.addPhysicalGroup(3, [vol_id], k+1)
    gmsh.model.setPhysicalName(3, k+1, "")

surf_tag_list = gmsh.model.occ.getEntities(2)

for k, surf_tag in enumerate(surf_tag_list):
    surf_id = surf_tag[1]
    gmsh.model.addPhysicalGroup(2, [surf_id], k+1)
    gmsh.model.setPhysicalName(2, k+1, "")

gmsh.model.addPhysicalGroup(1, [244], 1)
gmsh.model.setPhysicalName(1, 1, "")

#generate mesh
    
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.01)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.01)

gmsh.model.mesh.generate(3)

gmsh.write("panel_tube.vtk")      # for fenics
gmsh.write("panel_tube.msh")      # for fenics

mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model,
        comm,
        0,
        partitioner=dolfinx.mesh.create_cell_partitioner(
            dolfinx.mesh.GhostMode.shared_facet
        ),
    )

# facet_tags includes the tag of facets on bout boundaries.
gmsh.finalize()


ds_external = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
# 5: back surface, 7: top surface, 12 and 13 are spring areas
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags, metadata={"quadrature_degree": 4})

print('Read in data')

##########################
# import data
##########################
top_coor_load = np.loadtxt('top_surface_load.txt')
bot_coor_load = np.loadtxt('bot_surface_load.txt')

print(np.min(top_coor_load[:, 0]), np.max(top_coor_load[:, 0]))
print(np.min(top_coor_load[:, 1]), np.max(top_coor_load[:, 1]))
print(np.min(top_coor_load[:, 2]), np.max(top_coor_load[:, 2]))

print(np.min(bot_coor_load[:, 0]), np.max(bot_coor_load[:, 0]))
print(np.min(bot_coor_load[:, 1]), np.max(bot_coor_load[:, 1]))
print(np.min(bot_coor_load[:, 2]), np.max(bot_coor_load[:, 2]))

# the panel mesh generated here: x stars from 0 to length, y start from -panel_width/2 to panel_width/2, z starts from block_thick to panel_thick+block_thick
aa = (top_coor_load[:,1])/24.2*20 #-10 to 10
bb = (top_coor_load[:,0])/4.1*4.0 # -2 to 2
cc = top_coor_load[:,2]/0.1*panel_thick+panel_thick/2+block_thick  # block_thick to block_thick+panel_thick

top_coor_load[:,0] = aa
top_coor_load[:,1] = bb
top_coor_load[:,2] = cc

aaa = (bot_coor_load[:,1])/24.2*20
bbb = (bot_coor_load[:,0])/4.1*4.0
ccc = bot_coor_load[:,2]/0.1*panel_thick+panel_thick/2 + block_thick

bot_coor_load[:,0] = aaa
bot_coor_load[:,1] = bbb    # flip x and y, in pvade the x axis is the y axis in this code
bot_coor_load[:,2] = ccc

rotate_coor_x_top = top_coor_load[:,0]
rotate_coor_y_top = np.cos(stow_angle)*top_coor_load[:,1]-np.sin(stow_angle)*top_coor_load[:,2]
rotate_coor_z_top = np.sin(stow_angle)*top_coor_load[:,1]+np.cos(stow_angle)*top_coor_load[:,2]

rotate_coor_x_bot = bot_coor_load[:,0]
rotate_coor_y_bot = np.cos(stow_angle)*bot_coor_load[:,1]-np.sin(stow_angle)*bot_coor_load[:,2]
rotate_coor_z_bot = np.sin(stow_angle)*bot_coor_load[:,1]+np.cos(stow_angle)*bot_coor_load[:,2]

top_coor = np.zeros((np.shape(rotate_coor_x_top)[0], 3))
bot_coor = np.zeros((np.shape(rotate_coor_x_bot)[0], 3))

top_coor[:,0] = rotate_coor_x_top + 10
top_coor[:,1] = rotate_coor_y_top
top_coor[:,2] = rotate_coor_z_top
bot_coor[:,0] = rotate_coor_x_bot + 10
bot_coor[:,1] = rotate_coor_y_bot
bot_coor[:,2] = rotate_coor_z_bot

print(np.max(top_coor[:,0]))
print(np.min(top_coor[:,0]))
print(np.max(bot_coor[:,0]))
print(np.min(bot_coor[:,0]))

print(np.max(top_coor[:,1]))
print(np.min(top_coor[:,1]))
print(np.max(bot_coor[:,1]))
print(np.min(bot_coor[:,1]))

print(np.max(top_coor[:,2]))
print(np.min(top_coor[:,2]))
print(np.max(bot_coor[:,2]))
print(np.min(bot_coor[:,2]))

print(np.max(mesh.geometry.x[:,0]))
print(np.min(mesh.geometry.x[:,0]))
print(np.max(mesh.geometry.x[:,1]))
print(np.min(mesh.geometry.x[:,1]))
print(np.max(mesh.geometry.x[:,2]))
print(np.min(mesh.geometry.x[:,2]))

########################
# save data
########################

# xdmf = dolfinx.io.XDMFFile(mesh.comm, "displacement_with_real_tube.xdmf", "w")
# xdmf.write_mesh(mesh)

print('Create function space and functions')
# ================================================================
# Create all elements and function spaces
# ================================================================
P1 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
V = dolfinx.fem.FunctionSpace(mesh, P1)

fe_1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=1)
W = dolfinx.fem.FunctionSpace(mesh, fe_1)

# ================================================================
# Create all functions on function spaces
# ================================================================

# discplacement
v = ufl.TestFunction(V)
u = dolfinx.fem.Function(V, name="disp_with_real_tube")

t_vector = dolfinx.fem.Function(V, name="surface_tractions")

# ================================================================
# Create functions to do the structural mechanics calcs
# ================================================================

# # surface_load_z = dolfinx.fem.Constant(mesh, [0.0,0.0,1000.0])  # surface traction, N/m^2
# # surface_load_y = dolfinx.fem.Constant(mesh, [0.0,200.0,0.0])  # surface traction, N/m^2

def epsilon(u):
    return 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(u, lamda, mu):
    return lamda*ufl.nabla_div(u)*ufl.Identity(3) + 2*mu*epsilon(u)


# The deformation gradient, F = I + dy/dX
def F_(u):
    I = ufl.Identity(len(u))
    return I + ufl.grad(u)

# The Cauchy-Green deformation tensor, C = F.T * F
def C_(u):
    F = F_(u)
    return F.T * F


# Green–Lagrange strain tensor, E = 0.5*(C - I), green strain pairs with the second kirchhoff stress, they are all based on original configuration
def E_(u):
    I = ufl.Identity(len(u))
    C = C_(u)

    return 0.5 * (C - I)
    # return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

# The determinant of the deformation gradient, J = det(F)
def J_(u):
    F = F_(u)
    return ufl.det(F)

# The second Piola–Kirchhoff stress, S, it is symmetric
def S_(u, lamda, mu):
    E = E_(u)
    I = ufl.Identity(len(u))

    return lamda * ufl.tr(E) * I + 2.0 * mu * E
    # return lamda * ufl.tr(E) * I + 2.0 * mu * E

# The nominal stress tensor, P = P_.T
def P(u, lamda, mu):
    F = F_(u)
    S = S_(u, lamda, mu)
    return S*F.T


n = ufl.FacetNormal(mesh)

#================================================
# boundary condition
#================================================

# coordinates of all vertices
ver_coor = mesh.geometry.x   # array, shape: (number of nodes, 3)


#================================================
# get stress applied to each panel
#================================================

print('Interpolate the surface tractions')

top_surface_tags = np.arange(7,7+n_panel*7, 7).tolist()
bot_surface_tags = np.arange(5,5+n_panel*7, 7).tolist()

all_top_bot_surface_tag = np.array(top_surface_tags + bot_surface_tags)

block_tube_interface_tag1 = np.arange(74,74+n_panel*6, 6).tolist() 
block_tube_interface_tag2 = np.arange(75,75+n_panel*6, 6).tolist()

top_surface_facet_ID = []
bot_surface_facet_ID = []

# function_ty = dolfinx.fem.Function(W, name = 'ty')
# function_tz = dolfinx.fem.Function(W, name = 'tz')  

# convert load in normal and shear t0 xyz direction to: the load on x direction is not contributing to the normal and shear on y direction
# top_coor_load[:,3] is the surface traction along yita direction in the ppt
# top_coor_load[:,5] is the surface traction along normal direction fater rotation to stow angle
# top_coor_load[:,4] is the surface traction along x direction, x directio is not changed after rotating to stow angle


function_traction_1 = dolfinx.fem.Function(W, name = 't1') # along x, which is top_coor_load[:,4]
function_traction_2 = dolfinx.fem.Function(W, name = 't2') # along y, which is top_coor_load[:,3]
function_traction_3 = dolfinx.fem.Function(W, name = 't3') # along z, which is top_coor_load[:,5]



for j in range(n_panel):

    top_surface_facet_ID = facet_tags.find(top_surface_tags[j])
    bot_surface_facet_ID = facet_tags.find(bot_surface_tags[j])

    top_surface_ver_index = np.unique((dolfinx.cpp.mesh.entities_to_geometry(mesh, 2, np.array(top_surface_facet_ID, dtype=np.int32), False)).flatten())
    bot_surface_ver_index = np.unique((dolfinx.cpp.mesh.entities_to_geometry(mesh, 2, np.array(bot_surface_facet_ID, dtype=np.int32), False)).flatten())

    for i in top_surface_ver_index:
        # function_traction_1.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])] = griddata(top_coor,top_coor_load[:,4], [ver_coor[i,0]-j*panel_panel_gap, ver_coor[i,1], ver_coor[i,2]], method = 'nearest')[0]
        function_traction_2.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])] = griddata(top_coor,top_coor_load[:,3], [ver_coor[i,0]-j*panel_panel_gap, ver_coor[i,1], ver_coor[i,2]], method = 'nearest')[0]
        function_traction_3.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])] = griddata(top_coor,top_coor_load[:,5], [ver_coor[i,0]-j*panel_panel_gap, ver_coor[i,1], ver_coor[i,2]], method = 'nearest')[0]
    
        # function_ty.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])] = function_traction_2.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])]*np.cos(stow_angle) -\
        #                                                                       function_traction_3.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])]*np.sin(stow_angle)
        # function_tz.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])] = function_traction_2.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])]*np.sin(stow_angle) +\
        #                                                                       function_traction_3.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])]*np.cos(stow_angle)
    
    for k in bot_surface_ver_index:
        # function_traction_1.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])] = griddata(bot_coor,bot_coor_load[:,4], [ver_coor[k,0]-j*panel_panel_gap, ver_coor[k,1], ver_coor[k,2]], method = 'nearest')[0]
        function_traction_2.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])] = griddata(bot_coor,bot_coor_load[:,3], [ver_coor[k,0]-j*panel_panel_gap, ver_coor[k,1], ver_coor[k,2]], method = 'nearest')[0]
        function_traction_3.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])] = griddata(bot_coor,bot_coor_load[:,5], [ver_coor[k,0]-j*panel_panel_gap, ver_coor[k,1], ver_coor[k,2]], method = 'nearest')[0]
        
        # function_ty.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])] = -function_traction_2.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])]*np.cos(stow_angle) +\
        #                                                                       function_traction_3.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])]*np.sin(stow_angle)
        # function_tz.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])] = -function_traction_2.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])]*np.sin(stow_angle) -\
                                                                            #   function_traction_3.x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])]*np.cos(stow_angle)

#################################################
# verify the torque
#################################################
        
print('Verify the calculated torque')

class FillFunctionWithXCoords:
    def __init__(self):
        pass

    def __call__(self, x):
        coords = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)

        
        coords[0] = x[0]

        return coords

spatial_X_coords = dolfinx.fem.Function(W, name="spatial_coords_X")
spatial_X_coords.interpolate(FillFunctionWithXCoords())
    
class FillFunctionWithYCoords:
    def __init__(self):
        pass

    def __call__(self, x):
        coords = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)

        
        coords[0] = x[1]

        return coords

spatial_Y_coords = dolfinx.fem.Function(W, name="spatial_coords_Y")
spatial_Y_coords.interpolate(FillFunctionWithYCoords())

class FillFunctionWithZCoords:
    def __init__(self):
        pass

    def __call__(self, x):
        coords = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)

        
        coords[0] = x[2]
        

        return coords

spatial_Z_coords = dolfinx.fem.Function(W, name="spatial_coords_Z")
spatial_Z_coords.interpolate(FillFunctionWithZCoords())

T_p1 = []
T_p2 = []
T_q1 = []
T_q2 = []

F_y = []
F_z = []

for i in range(n_panel):

    # T_p1.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Y_coords+block_thick*np.sin(stow_angle), function_tz)*ds_external(5+i*7))) \
    #         + dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Y_coords+(block_thick+panel_thick)*np.sin(stow_angle), function_tz)*ds_external(7+i*7))))
        
    # T_p2.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(block_thick*np.sin(stow_angle), function_tz)*ds_external(5+i*7))) \
    #     - dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner((block_thick+panel_thick)*np.sin(stow_angle), function_tz)*ds_external(7+i*7))))
    
    # cc_c1.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_tz)*ds_external(5+i*7))))
    # cc_c2.append(- dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_tz)*ds_external(7+i*7))))
    
    # T_q1.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Z_coords-block_thick*np.cos(stow_angle), function_ty)*ds_external(5+i*7))) \
    #     - dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Z_coords-(block_thick+panel_thick)*np.cos(stow_angle), function_ty)*ds_external(7+i*7))))             # torque applied to panel

    # T_q2.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(block_thick*np.cos(stow_angle), function_ty)*ds_external(5+i*7))) \
    #     - dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner((block_thick+panel_thick)*np.cos(stow_angle), function_ty)*ds_external(7+i*7))))             # torque applied to panel

    # dd_c1.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_ty)*ds_external(5+i*7))))             # torque applied to panel
    # dd_c2.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_ty)*ds_external(7+i*7))))

    T_p1.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Y_coords+block_thick*np.sin(stow_angle), function_traction_3)*ds_external(5+i*7))) \
            + dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Y_coords+(block_thick+panel_thick)*np.sin(stow_angle), function_traction_3)*ds_external(7+i*7))))
        
    T_p2.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(block_thick*np.sin(stow_angle), function_traction_3)*ds_external(5+i*7))) \
        - dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner((block_thick+panel_thick)*np.sin(stow_angle), function_traction_3)*ds_external(7+i*7))))
    
    F_z.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_traction_3)*ds_external(5+i*7))) + dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_traction_3)*ds_external(7+i*7))))
    
    T_q1.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Z_coords-block_thick*np.cos(stow_angle), function_traction_2)*ds_external(5+i*7))) \
        - dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Z_coords-(block_thick+panel_thick)*np.cos(stow_angle), function_traction_2)*ds_external(7+i*7))))             # torque applied to panel

    T_q2.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(block_thick*np.cos(stow_angle), function_traction_2)*ds_external(5+i*7))) \
        - dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner((block_thick+panel_thick)*np.cos(stow_angle), function_traction_2)*ds_external(7+i*7))))             # torque applied to panel

    F_y.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_traction_2)*ds_external(5+i*7))) + dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_traction_2)*ds_external(7+i*7))))            
    
total_torque = np.array(T_p1)+np.array(T_p2)+np.array(T_q1)+np.array(T_q2)  # undeformed configuration

print('torque applied by each panel:', total_torque)
print('drag force on each panel:', F_y)
print('lift force on each panel:', F_z)



# rotation of connector
x_panel = np.arange(10)*(panel_length+panel_panel_gap) + panel_length/2# location of panles

phi = []

for i in range(n_panel):
    # phi_i = 0
    # for k in range(i, n_panel):
    #     phi_i += x_panel[i]/G/Ip*total_torque[k] # on undeformed configuration
    # for j in range(i):
    #     phi_i = phi_i + total_torque[j]*x_panel[j]/G/Ip
    # phi.append(phi_i)

    phi_i = 0
    for k in range(i, n_panel):
        phi_i += x_panel[i]/G/Ip*(T_p1[k] + T_p2[k] + T_q1[k] + T_q2[k]) # on undeformed configuration
    for j in range(i):
        phi_i = phi_i + (T_p1[j] + T_p2[j] + T_q1[j] + T_q2[j])*x_panel[j]/G/Ip
    phi.append(phi_i)




    # deflection of the beam, statically indeterminate structure!!

    #               |  |  |    |   |   |        |     |    |  |   |   |              panel load (i)
    #       fixed|------------------------------------------------------
    #                                                                   |  supports (1)

    # equivalent structure:
    #               |  |  |       |  |  |        |   |  |     |  |  |              panel load (i)
    #       fixed|------------------------------------------------------
    #                                                                   |  loads (1)

    # delta_1i is the displacement at location 1, caused by a unit load applied at i
deflection_y = []
deflection_z = []

delta_11 = ((total_length)**3)/E_tube/Iz/3.0

delta_1i_list = []
for i in range(n_panel):  
    delta_1i = x_panel[i]**2/6.0/E_tube/Iz*(3*total_length-x_panel[i])
    delta_1i_list.append(delta_1i)

reaction_force_right_support_z = -np.sum(np.array(delta_1i_list)*np.array(F_z))/delta_11
reaction_force_right_support_y = -np.sum(np.array(delta_1i_list)*np.array(F_y))/delta_11

delta_force = np.zeros((n_panel+1, n_panel+1)) # ij component is the deflection at ith location cause by unit load applied at jth location

i_location_list = []

for i in range(n_panel+1):  # panels
    if i != n_panel:
        ith_location = (panel_length+panel_panel_gap)*i + panel_length/2.0
    else:
        ith_location = total_length
    i_location_list.append(ith_location)
    for j in range(n_panel+1): # load location
        
        if j != n_panel:
            jth_location = (panel_length+panel_panel_gap)*j + panel_length/2.0
        else:
            jth_location = total_length
        if j!= n_panel:
            if i < j:
                delta_force[i, j] = ith_location**2/6.0/E_tube/Iz*(3*jth_location-ith_location)
            else:
                delta_force[i, j] = jth_location**2/6.0/E_tube/Iz*(3*ith_location-jth_location)
        if j == n_panel:
            delta_force[i, j] = ith_location**2/6.0/E_tube/Iz*(3*total_length-ith_location)



deflection_z = np.dot(delta_force, np.array(F_z+[reaction_force_right_support_z]).T)     
deflection_y = np.dot(delta_force, np.array(F_y+[reaction_force_right_support_y]).T)          

# print(deflection_y)
# print(deflection_z)

# fig1 = plt.figure()
# plt.plot([0]+i_location_list, np.concatenate([[0],deflection_z])*100, '-o')
# plt.xlabel('tube axis')
# plt.ylabel('Deflection along z (cm)')
# plt.grid()
# fig2 = plt.figure()
# plt.plot([0]+i_location_list, np.concatenate([[0],deflection_y])*100, '-o')
# plt.xlabel('tube axis')
# plt.ylabel('Deflection along y (cm)')
# plt.grid()
# plt.show()


print('Define BCs')
###################################
# apply BC
###################################
# fixed left tube surface
bcs = []
left_tube_entities = facet_tags.find(133)
left_tube_dofs = dolfinx.fem.locate_dofs_topological(V, 2, left_tube_entities)
left_disp_value = dolfinx.fem.Constant(mesh, PETSc.ScalarType([0,0,0]))
bcs.append(dolfinx.fem.dirichletbc(left_disp_value, left_tube_dofs, V))    

points = mesh.geometry.x

np.savetxt('nodes_coor.txt', points)
num_nodes = np.shape(points)[0]
# for center line:
center_line_index = []
right_center_point = []
tube_line_point = []



for i in range(num_nodes):
    if abs(points[i, 1]) < 1.0e-10 and abs(points[i, 2])<1.0e-10:
        center_line_index.append(i)
    if abs(points[i, 1]) < 1.0e-10 and abs(points[i, 2])<1.0e-10 and abs(points[i, 0]-total_length) < 1.0e-14:
        right_center_point.append(i)
    if abs(points[i, 1]+0.03420201433256685186) < 1.0e-8 and abs(points[i, 2]-0.09396926207859085389)<1.0e-8:
        tube_line_point.append(i)


tube_line_dof_uz = dolfinx.fem.locate_dofs_topological(V.sub(2), 0, tube_line_point)
tube_line_dof_x = dolfinx.fem.locate_dofs_topological(W, 0, tube_line_point)



# center_line_dof = dolfinx.fem.locate_dofs_topological(V, 0, center_line_index)
# bcs.append(dolfinx.fem.dirichletbc(left_disp_value, center_line_dof, V))  

# # for right surface center point, uz=uy=0
# right_center_line_dof_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, right_center_point)
# right_center_line_dof_z = dolfinx.fem.locate_dofs_topological(V.sub(2), 0, right_center_point)

# print(right_center_line_dof_y, right_center_line_dof_z)

# bcs.append(dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 0.0), right_center_line_dof_y, V.sub(1)))
# bcs.append(dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 0.0), right_center_line_dof_z, V.sub(2)))
        
# for center line, uz=uy=0
center_line_dof_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, center_line_index)
center_line_dof_z = dolfinx.fem.locate_dofs_topological(V.sub(2), 0, center_line_index)

print(center_line_dof_y, center_line_dof_z)

bcs.append(dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 0.0), center_line_dof_y, V.sub(1)))
bcs.append(dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 0.0), center_line_dof_z, V.sub(2)))


################################################
# define variational form and solve 
################################################
    
print('Define variational form and solve')



    
t_vector.sub(1).interpolate(function_traction_2)
t_vector.sub(2).interpolate(function_traction_3)

res = 0

for i in range(n_panel):
    i_panel = 1+2*i
    i_block = 2+2*i
    res = res - ufl.inner(P(u, lamda_panel, mu_panel), ufl.nabla_grad(v))*dx(i_panel) - ufl.inner(P(u, lamda_block, mu_block), ufl.nabla_grad(v))*dx(i_block)
for i in all_top_bot_surface_tag:
    res = res  + ufl.inner(t_vector, v)*ds_external(i)

res = res  - ufl.inner(sigma(u, lamda_tube, mu_tube), ufl.nabla_grad(v))*dx(21)

print('     solve')
print('          solve 1')
nonlinear_problem = dolfinx.fem.petsc.NonlinearProblem(res, u, bcs=bcs)
print('          solve 2')
solver = dolfinx.nls.petsc.NewtonSolver(comm, nonlinear_problem)

solver.atol = 1e-8
solver.rtol = 1e-8
# solver.relaxation_parameter = 0.5
# solver.max_it = 500
# solver.convergence_criterion = "residual"
solver.convergence_criterion = "incremental"

# We can customize the linear solver used inside the NewtonSolver by
# modifying the PETSc options
print('          solve 3')
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()

print('          solve 4')

opts[f"{option_prefix}ksp_type"] = "preonly"
# opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_type"] = "ksp"

print('          solve 5')
solver.solve(u)

print(u.x.array[center_line_dof_z])
print(u.x.array[center_line_dof_y])

# xdmf = dolfinx.io.XDMFFile(mesh.comm, "displacement_with_real_tube.xdmf", "w")
# xdmf.write_mesh(mesh)
# xdmf.write_function(u, 0)
# xdmf.close()

# print(u.x.array[center_line_dof*3+1])
# print(u.x.array[center_line_dof*3+2])
# print(u.x.array[center_line_dof*3])

# print()
# exit()

print('Post process')

################################
# for post processing
################################


def right_bottom_line(x):
    return np.logical_and(np.isclose(x[1], block_width/2*np.cos(stow_angle)-block_thick*np.sin(stow_angle)), np.isclose(x[2], block_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle)))
    # return np.logical_and(np.isclose(x[1], panel_width/2*np.cos(stow_angle)-block_thick*np.sin(stow_angle)), np.isclose(x[2], panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle)))


phi_predict = []

for i in range(n_panel):
    submesh, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(mesh, 3, cell_tags.find(2+2*i)) # submesh of each block

    # finite element used for velocity saving
    disp_save = ufl.VectorElement("Lagrange", submesh.ufl_cell(), degree=1)
       
    # Make function spaces for each of these subdomains
    disp_save_fs = dolfinx.fem.FunctionSpace(submesh, disp_save)
    disp_save_fn = dolfinx.fem.Function(disp_save_fs, name="disp")
    disp_save_fn.interpolate(u)

    right_bottom_line_index = dolfinx.mesh.locate_entities_boundary(submesh, 1, right_bottom_line)

    right_bottom_line_dof_uz = dolfinx.fem.locate_dofs_topological(disp_save_fs.sub(2), 1, right_bottom_line_index)

    # right_bottom_line_dof_uz = (right_bottom_line_dof*3+2)

    uz_right = disp_save_fn.x.array[right_bottom_line_dof_uz]

    phi_predict.append(np.mean(np.arcsin((block_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle)+uz_right)/(((block_length/2.0)**2+block_thick**2)**0.5)))-np.arcsin((block_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle))/(((block_length/2.0)**2+block_thick**2)**0.5)))
    # phi_predict.append(np.mean(np.arcsin((panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle)+uz_right)/(((panel_length/2.0)**2+block_thick**2)**0.5)))-np.arcsin((panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle))/(((panel_length/2.0)**2+block_thick**2)**0.5)))

    with dolfinx.io.XDMFFile(comm, "block_disp_solution"+str(i)+".xdmf", "w") as fp:
        fp.write_mesh(submesh)
        fp.write_function(disp_save_fn, 0.0)

submesh, entity_map, vertex_map, geom_map = dolfinx.mesh.create_submesh(mesh, 3, cell_tags.find(21))
disp_save_tube = ufl.VectorElement("Lagrange", submesh.ufl_cell(), degree=1)
       
# Make function spaces for each of these subdomains
disp_save_tube_fs = dolfinx.fem.FunctionSpace(submesh, disp_save_tube)
disp_save_tube_fn = dolfinx.fem.Function(disp_save_tube_fs, name="save_disp")
disp_save_tube_fn.interpolate(u)

with dolfinx.io.XDMFFile(comm, "panel_disp_solution_tube.xdmf", "w") as fp:
    fp.write_mesh(submesh)
    fp.write_function(disp_save_tube_fn, 0.0)



# print(np.arcsin((panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle)-0.19)/l_0_prime))

# print(np.array(phi_predict)/np.array(phi))

# # print(tube_torque_angle)
# print(np.array(ini_angle)/2/np.pi*360)
# print(np.array(fin_angle)/2/np.pi*360)

# print(np.asarray(phi_predict)/2/np.pi*360.0)
    
print(np.array(phi_predict)/np.array(phi))

# print((np.arcsin((panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle)-0.014)/(((panel_length/2.0)**2+block_thick**2)**0.5))-np.arcsin((panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle))/(((panel_length/2.0)**2+block_thick**2)**0.5)))/2/np.pi*360)

plt.plot(x_panel, np.asarray(phi)/2/np.pi*360.0, 'bo', label='Theoretical')
plt.plot(x_panel, -np.asarray(phi_predict)/2/np.pi*360.0, 'r*', label='With Tube (from block)')
# plt.plot(spatial_X_coords.x.array[tube_line_dof_x], -np.arcsin(u.x.array[tube_line_dof_uz]/r_out)/2/np.pi*360, label='With Tube (from tube)')
# plt.plot(x_panel, -np.asarray(phi_predict)*np.sin(np.array(phi)+stow_angle)/np.sin(stow_angle)/2/np.pi*360.0, 'r*', label='With Tube (from block)')

plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('Rotation Angle (degree)')
plt.show()









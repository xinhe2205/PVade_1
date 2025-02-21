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

# ================================================================
# Declare all constants
# ================================================================

# Define structural properties
E_Y = 1.4e18  # Value from CSM3, turek-hron, youngs modulus is set suoer high to get rid of elastic deformation
nu = 0.3  # Value from CSM3, turek-hron
mu = E_Y / (2.0 * (1.0 + nu))
lamda = nu*E_Y/(1+nu)/(1-2*nu)  # Value from CSM3, turek-hron

G = 77.0*10**9 # shear modulus of tortional tube

r_out = 0.1  # out radius of tube
r_in = 0.09 # inner radius of tube

Ip = np.pi/2*(r_out**4-r_in**4) # moment of inertia of tube

panel_length = 2.0   
panel_width = 4.0
panel_thick = 0.1

block_thick = r_out

block_length = 0.1*panel_length
block_width = block_length

stow_angle = np.pi/9 # theta_i

l_0_prime = ((panel_width/2)**2+block_thick**2)**0.5

theta_prime = np.arcsin((panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle))/l_0_prime)

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0) 
gmsh.model.add("panel_with_springs")

panel = gmsh.model.occ.add_box(-panel_length/2, -panel_width/2, block_thick, panel_length, panel_width, panel_thick)
block = gmsh.model.occ.add_box(-block_length/2, -block_width/2, 0, block_length, block_width, block_thick)


fixed_point1 = gmsh.model.occ.add_point(-block_length/2, 0, 0)
fixed_point2 = gmsh.model.occ.add_point(block_length/2, 0, 0)

fixed_line = gmsh.model.occ.add_line(fixed_point1, fixed_point2)

spring_area_1 = gmsh.model.occ.add_rectangle(-block_length/2, -block_width/2, 0, block_length, block_width/2)
spring_area_2 = gmsh.model.occ.add_rectangle(-block_length/2, 0, 0, block_length, block_width/2)

# front_area = gmsh.model.occ.add_rectangle(-panel_length/2, 0, panel_thick+block_thick, panel_length, panel_width/2) 
# split the front area, half of the area applied f, the other half is applied -f, so it introduce a torque.

domain = gmsh.model.occ.fragment([(3, panel), (3, block)], [(0, fixed_point1),(0, fixed_point2),(1, fixed_line), (2, spring_area_1), (2, spring_area_2)])

domain = gmsh.model.occ.rotate([(3,1), (3,2)], 0,0,0,1,0,0,stow_angle)

gmsh.model.occ.synchronize()
gmsh.write("panel_springs.brep")



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

#generate mesh
    
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.05)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.05)

gmsh.model.mesh.generate(3)

gmsh.write("panel_springs.vtk")      # for fenics
gmsh.write("panel_springs.msh")      # for fenics

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
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
##########################
# import data
##########################
top_coor_load = np.loadtxt('top_surface_load.txt')
bot_coor_load = np.loadtxt('bot_surface_load.txt')



aa = (top_coor_load[:,1])/24.2*20.0 #-10 to 10
bb = (top_coor_load[:,0])/4.1*4.0 # -2 to 2
cc = top_coor_load[:,2]+0.05+block_thick  # block_thick to block_thick+panel_thick

top_coor_load[:,0] = aa
top_coor_load[:,1] = bb
top_coor_load[:,2] = cc

aaa = (bot_coor_load[:,1])/24.2*20.0
bbb = (bot_coor_load[:,0])/4.1*4.0
ccc = bot_coor_load[:,2]+0.05 + block_thick

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

# print(np.max(top_coor[:,0]))
# print(np.min(top_coor[:,0]))
# print(np.max(bot_coor[:,0]))
# print(np.min(bot_coor[:,0]))

# print(np.max(top_coor[:,1]))
# print(np.min(top_coor[:,1]))
# print(np.max(bot_coor[:,1]))
# print(np.min(bot_coor[:,1]))

# print(np.max(top_coor[:,2]))
# print(np.min(top_coor[:,2]))
# print(np.max(bot_coor[:,2]))
# print(np.min(bot_coor[:,2]))

# print(np.max(mesh.geometry.x[:,0]))
# print(np.min(mesh.geometry.x[:,0]))
# print(np.max(mesh.geometry.x[:,1]))
# print(np.min(mesh.geometry.x[:,1]))
# print(np.max(mesh.geometry.x[:,2]))
# print(np.min(mesh.geometry.x[:,2]))
# exit()




########################
# save data
########################

xdmf = dolfinx.io.XDMFFile(mesh.comm, "displacement.xdmf", "w")
xdmf.write_mesh(mesh)

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
u = dolfinx.fem.Function(V)

t_vector = dolfinx.fem.Function(V)

# ================================================================
# Create functions to do the structural mechanics calcs
# ================================================================

# surface_load_z = dolfinx.fem.Constant(mesh, [0.0,0.0,1000.0])  # surface traction, N/m^2
# surface_load_y = dolfinx.fem.Constant(mesh, [0.0,200.0,0.0])  # surface traction, N/m^2


z_unit_vector = dolfinx.fem.Constant(mesh, [0.0,0.0,1.0])  # surface traction, N/m^2


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
def S_(u):
    E = E_(u)
    I = ufl.Identity(len(u))

    return lamda * ufl.tr(E) * I + 2.0 * mu * E
    # TODO: Why does the above form give a better result and where does it come from?

    # return lamda * ufl.tr(E) * I + 2.0 * mu * E

# The nominal stress tensor, P = P_.T
def P(u):
    F = F_(u)
    S = S_(u)
    return S*F.T


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

n = ufl.FacetNormal(mesh)

#================================================
# boundary condition
#================================================

# fixed a line
def clamped_boundary(x):
    return np.logical_and(np.isclose(x[1], 0), np.isclose(x[2], 0))

boundary_line = dolfinx.mesh.locate_entities_boundary(mesh, 1, clamped_boundary)


u_D = np.array([0.0, 0.0, 0.0])
bc1 = dolfinx.fem.dirichletbc(u_D, dolfinx.fem.locate_dofs_topological(V, 1, boundary_line), V)

bcs = [bc1]

#================================================
# get stress applied to each panel
#================================================

N = 10 # number of panels

# coordinates of all vertices
ver_coor = mesh.geometry.x   # array, shape: (number of nodes, 3)

top_surface_facet_ID = facet_tags.find(7)
bot_surface_facet_ID = facet_tags.find(5)

top_surface_ver_index = np.unique((dolfinx.cpp.mesh.entities_to_geometry(mesh, 2, top_surface_facet_ID, False)).flatten())
bot_surface_ver_index = np.unique((dolfinx.cpp.mesh.entities_to_geometry(mesh, 2, bot_surface_facet_ID, False)).flatten())

function_ty = {}
function_tz = {}    # to call function: function_ty["ty_panel_1"]
for j in range(N):
    ty_name = f"ty_panel_{j}"
    tz_name = f"tz_panel_{j}"
    function_ty[ty_name] = dolfinx.fem.Function(W, name=ty_name)
    function_tz[tz_name] = dolfinx.fem.Function(W, name=tz_name)
    for i in top_surface_ver_index:
        function_ty[ty_name].x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])] = griddata(top_coor,top_coor_load[:,3], [ver_coor[i,0]+j*2+1, ver_coor[i,1], ver_coor[i,2]], method = 'nearest')[0]
        function_tz[tz_name].x.array[dolfinx.fem.locate_dofs_topological(W, 0, [i])] = griddata(top_coor,top_coor_load[:,5], [ver_coor[i,0]+j*2+1, ver_coor[i,1], ver_coor[i,2]], method = 'nearest')[0]
    for k in bot_surface_ver_index:
        function_ty[ty_name].x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])] = griddata(bot_coor,bot_coor_load[:,3], [ver_coor[k,0]+j*2+1, ver_coor[k,1], ver_coor[k,2]], method = 'nearest')[0]
        function_tz[tz_name].x.array[dolfinx.fem.locate_dofs_topological(W, 0, [k])] = griddata(bot_coor,bot_coor_load[:,5], [ver_coor[k,0]+j*2+1, ver_coor[k,1], ver_coor[k,2]], method = 'nearest')[0]

#================================================
# get spring stiffness
#================================================



x_panel = np.arange(10)*panel_length + 1# location of panles

T_p1 = []
T_p2 = []
T_q1 = []
T_q2 = []
phi = []
K = []

cc_c1 = []
dd_c1 = []
cc_c2 = []
dd_c2 = []

for i in range(N):

    T_p1.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Y_coords+block_thick*np.sin(stow_angle), function_tz[f"tz_panel_{i}"])*ds_external(5))) \
        + dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Y_coords+(block_thick+panel_thick)*np.sin(stow_angle), function_tz[f"tz_panel_{i}"])*ds_external(7))))
    
    T_p2.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(block_thick*np.sin(stow_angle), function_tz[f"tz_panel_{i}"])*ds_external(5))) \
        - dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner((block_thick+panel_thick)*np.sin(stow_angle), function_tz[f"tz_panel_{i}"])*ds_external(7))))

    cc_c1.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_tz[f"tz_panel_{i}"])*ds_external(5))))
    cc_c2.append(- dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_tz[f"tz_panel_{i}"])*ds_external(7))))


    T_q1.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Z_coords-block_thick*np.cos(stow_angle), function_ty[f"ty_panel_{i}"])*ds_external(5))) \
        - dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Z_coords-(block_thick+panel_thick)*np.cos(stow_angle), function_ty[f"ty_panel_{i}"])*ds_external(7))))             # torque applied to panel

    T_q2.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(block_thick*np.cos(stow_angle), function_ty[f"ty_panel_{i}"])*ds_external(5))) \
        - dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner((block_thick+panel_thick)*np.cos(stow_angle), function_ty[f"ty_panel_{i}"])*ds_external(7))))             # torque applied to panel
    
    dd_c1.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_ty[f"ty_panel_{i}"])*ds_external(5))))             # torque applied to panel
    dd_c2.append(- dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(1, function_ty[f"ty_panel_{i}"])*ds_external(7))))

print(T_p1)
print(T_p2)

print(T_q1)
print(T_q2)

print(cc_c1)
print(cc_c2)
print(dd_c1)
print(dd_c2)
print(np.array(T_p1)+np.array(T_p2)+np.array(T_q1)+np.array(T_q2))

# exit()

for i in range(N):
    phi_i = 0
    for k in range(i, N):
        phi_i = x_panel[i]/G/Ip*(T_p1[k] + T_p2[k] + T_q1[k] + T_q2[k]) # on undeformed configuration
    for j in range(i):
        phi_i = phi_i + (T_p1[j] + T_p2[j] + T_q1[j] + T_q2[j])*x_panel[j]/G/Ip
    phi.append(phi_i)

    K_i = 12*((T_p1[i]+T_q2[i])*(np.cos(stow_angle+phi_i)/np.cos(stow_angle)) + (T_q1[i]+T_p2[i]) *(np.sin(stow_angle+phi_i)/np.sin(stow_angle)))/((block_length)**3)/np.cos(stow_angle+phi_i)/(np.sin(stow_angle+phi_i)-np.sin(stow_angle))/block_width

    K.append(K_i)


print(np.array(phi)/2/np.pi*360)

# exit()
# for i in range(N):

#     T_p.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Y_coords, function_tz[f"tz_panel_{j}"])*ds_external(5))) \
#         + dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Y_coords, function_tz[f"tz_panel_{j}"])*ds_external(7))))

#     T_q.append(-dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Z_coords, function_ty[f"ty_panel_{j}"])*ds_external(5))) \
#         - dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Z_coords, function_ty[f"ty_panel_{j}"])*ds_external(7))))             # torque applied to panel



# for i in range(N):
#     phi_i = 0
#     for k in range(i, N):
#         phi_i = x_panel[i]/G/Ip*(T_p[k] + T_q[k])
#     for j in range(i):
#         phi_i = phi_i + (T_p[j] + T_q[j])*x_panel[j]/G/Ip
#     phi.append(phi_i)

#     K_i = 12*(T_p[i]*(np.cos(stow_angle+phi_i)/np.cos(stow_angle)) + T_q[i] *(np.sin(stow_angle+phi_i)/np.sin(stow_angle)))/((block_length)**3)/np.cos(stow_angle+phi_i)/(np.sin(stow_angle+phi_i)-np.sin(stow_angle))/block_width
#     K.append(K_i)

################################
# for post processing
################################

# get the dof at the right bottom edge and left bottom edge used to calculate the rotation of panel
def right_bottom_line(x):
    return np.logical_and(np.isclose(x[1], panel_width/2*np.cos(stow_angle)-block_thick*np.sin(stow_angle)), np.isclose(x[2], panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle)))

def left_bottom_line(x):
    return np.logical_and(np.isclose(x[1], -panel_width/2*np.cos(stow_angle)-block_thick*np.sin(stow_angle)), np.isclose(x[2], -panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle)))

right_bottom_line_index = dolfinx.mesh.locate_entities_boundary(mesh, 1, right_bottom_line)
left_bottom_line_index = dolfinx.mesh.locate_entities_boundary(mesh, 1, left_bottom_line)

right_bottom_line_dof = dolfinx.fem.locate_dofs_topological(V, 1, right_bottom_line_index)
left_bottom_line_dof = dolfinx.fem.locate_dofs_topological(V, 1, left_bottom_line_index)

right_bottom_line_dof_uz = (right_bottom_line_dof*3+2)
left_bottom_line_dof_uz = (left_bottom_line_dof*3+2)

phi_predict = []


################################################
# define variational form and solve 
################################################
for i in range(N):
    print(i)
    
    K_vector = dolfinx.fem.Constant(mesh, [0.0,0.0,K[i]])  # surface traction, N/m^2
    t_vector.sub(1).interpolate(function_ty[f"ty_panel_{i}"])
    t_vector.sub(2).interpolate(function_tz[f"tz_panel_{i}"])

    
    res = -ufl.inner(P(u), ufl.nabla_grad(v))*dx \
    -ufl.inner(K_vector, u)*ufl.inner(z_unit_vector, v)*ds_external(13) \
    -ufl.inner(K_vector, u)*ufl.inner(z_unit_vector, v)*ds_external(12) \
    +ufl.inner(t_vector, v)*ds_external(7) \
    +ufl.inner(t_vector, v)*ds_external(5)
    
    
    nonlinear_problem = dolfinx.fem.petsc.NonlinearProblem(res, u, bcs=bcs)
    solver = dolfinx.nls.petsc.NewtonSolver(comm, nonlinear_problem)

    solver.atol = 1e-8
    solver.rtol = 1e-8
    # solver.relaxation_parameter = 0.5
    # solver.max_it = 500
    # solver.convergence_criterion = "residual"
    solver.convergence_criterion = "incremental"

    # We can customize the linear solver used inside the NewtonSolver by
    # modifying the PETSc options
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()

    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"


    solver.solve(u)

    xdmf.write_function(u, i)


    uz_right = u.x.array[right_bottom_line_dof_uz]

    print(uz_right)

    phi_predict.append(np.mean(np.arcsin((panel_width/2*np.sin(stow_angle)+block_thick*np.cos(stow_angle)+uz_right)/l_0_prime))-theta_prime)

    # uz_left = u.x.array[left_bottom_line_dof_uz]

    # angle_z1.append(np.mean(np.arcsin(uz_left/panel_width*2)))
xdmf.close()

print(phi_predict)

plt.plot(range(1,1+N), np.asarray(phi_predict)/2/np.pi*360.0, 'bo', label='With Springs')
plt.plot(range(1,1+N), np.asarray(phi)/2/np.pi*360.0, 'r*', label='With Tube')
plt.grid()
plt.legend()
plt.xlabel('Panel ID')
plt.ylabel('Rotation Angle (degree)')
plt.show()

plt.plot(range(1,1+N), (np.asarray(phi_predict)-np.asarray(phi))/np.asarray(phi), 'bo')
plt.grid()
plt.legend()
plt.xlabel('Panel ID')
plt.ylabel('error')
plt.show()













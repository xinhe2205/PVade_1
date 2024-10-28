import numpy as np
import gmsh
import dolfinx
import ufl
from mpi4py import MPI
from numba import jit
from petsc4py import PETSc
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD

panel_length = 2.0   # di
panel_width = 4.0
panel_thick = 0.2

spring_area_width = 0.05   # wi

spring_to_center_distance = 0.2

stow_angle = np.pi/6 # theta_i

gmsh.initialize()
gmsh.model.add("panel_with_springs")

panel = gmsh.model.occ.add_box(-panel_length/2, -panel_width/2, 0, panel_length, panel_width, panel_thick)

fixed_point1 = gmsh.model.occ.add_point(-panel_length/2, 0, 0)
fixed_point2 = gmsh.model.occ.add_point(panel_length/2, 0, 0)

fixed_line = gmsh.model.occ.add_line(fixed_point1, fixed_point2)

spring_area_1 = gmsh.model.occ.add_rectangle(-panel_length/2, -spring_to_center_distance-spring_area_width/2, 0, panel_length, spring_area_width)
spring_area_2 = gmsh.model.occ.add_rectangle(-panel_length/2, spring_to_center_distance-spring_area_width/2, 0, panel_length, spring_area_width)

front_area = gmsh.model.occ.add_rectangle(-panel_length/2, 0, panel_thick, panel_length, panel_width/2) 
# split the front area, half of the area applied f, the other half is applied -f, so it introduce a torque.

domain = gmsh.model.occ.fragment([(3, panel)], [(0, fixed_point1),(0, fixed_point2),(1, fixed_line), (2, spring_area_1), (2, spring_area_2), (2, front_area)])
# the tag of fixed line is not changed.

spring_area_1_tag = domain[1][4][0][1]
spring_area_2_tag = domain[1][5][0][1]

fixed_line_tag = fixed_line

domain = gmsh.model.occ.rotate([(3,1)], 0,0,0,1,0,0,stow_angle)

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
    
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.1)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.1)

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
# 9 and 6 are spring areas, 11 and 12 on front surfaced, 11 bottom, 12 top part
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

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

# ================================================================
# Create all functions on function spaces
# ================================================================

# discplacement
v = ufl.TestFunction(V)
u = dolfinx.fem.Function(V)

# ================================================================
# Declare all constants
# ================================================================

# Define structural properties
E_Y = dolfinx.fem.Constant(mesh, 1.4e15)  # Value from CSM3, turek-hron, youngs modulus is set suoer high to get rid of elastic deformation
nu = dolfinx.fem.Constant(mesh, 0.3)  # Value from CSM3, turek-hron
mu = E_Y / (2.0 * (1.0 + nu))
lamda = nu*E_Y/(1+nu)/(1-2*nu)  # Value from CSM3, turek-hron

G = 77.0*10**9 # shear modulus of tortional tube

r_out = 0.1  # out radius of tube
r_in = 0.09 # inner radius of tube

Ip = np.pi/2*(r_out**4-r_in**4) # moment of inertia of tube

# ================================================================
# Create functions to do the structural mechanics calcs
# ================================================================

surface_load = dolfinx.fem.Constant(mesh, [0.0,0.0,1000.0])  # surface traction, N/m^2

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
        coords = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)

        coords[0] = 0.0
        coords[1] = 0.0
        coords[2] = x[1]

        return coords

spatial_Y_coords = dolfinx.fem.Function(V, name="spatial_coords")
spatial_Y_coords.interpolate(FillFunctionWithYCoords())

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
# get spring stiffness
#================================================

N = 10 # number of panels

x_panel = np.arange(1,11)*panel_length# location of panles

T_half = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(spatial_Y_coords, surface_load)*ds_external(12)))# torque applied to panel
T = 2*T_half

phi = []
K = []
for i in range(N):
    phi_i = x_panel[i]/G/Ip*T*(N-i)
    for j in range(i):
        phi_i = phi_i + T*x_panel[j]/G/Ip
    phi.append(phi_i)
    K_i = 12*T/((spring_to_center_distance*2+spring_area_width)**3-(spring_to_center_distance*2-spring_area_width)**3)/np.cos(stow_angle+phi_i)/(np.sin(stow_angle+phi_i)-np.sin(stow_angle))/panel_length*np.cos(stow_angle+phi_i)/np.cos(stow_angle)
    K.append(K_i)

print(K)

################################
# for post processing
################################

# get the dof at the right bottom edge and left bottom edge used to calculate the rotation of panel
def right_bottom_line(x):
    return np.logical_and(np.isclose(x[1], panel_width/2*np.cos(stow_angle)), np.isclose(x[2], panel_width/2*np.sin(stow_angle)))

def left_bottom_line(x):
    return np.logical_and(np.isclose(x[1], -panel_width/2*np.cos(stow_angle)), np.isclose(x[2], -panel_width/2*np.sin(stow_angle)))

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
    
    res = -ufl.inner(P(u), ufl.nabla_grad(v))*dx-ufl.inner(K_vector, u)*ufl.inner(z_unit_vector, v)*ds_external(6) \
    -ufl.inner(K_vector, u)*ufl.inner(z_unit_vector, v)*ds_external(9) \
    -ufl.inner(surface_load, v)*ds_external(11)+ufl.inner(surface_load, v)*ds_external(12)
    
    
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

    phi_predict.append(np.mean(np.arcsin((uz_right+panel_width/2*np.sin(stow_angle))/panel_width*2))-stow_angle)

    # uz_left = u.x.array[left_bottom_line_dof_uz]

    # angle_z1.append(np.mean(np.arcsin(uz_left/panel_width*2)))
xdmf.close()

print(np.abs((np.asarray(phi_predict)-np.asarray(phi))/np.asarray(phi)))

plt.plot(range(1,1+N), np.asarray(phi_predict)/2/np.pi*360.0, 'bo', label='With Springs')
plt.plot(range(1,1+N), np.asarray(phi)/2/np.pi*360.0, 'r*', label='With Tube')
plt.grid()
plt.legend()
plt.xlabel('Panel ID')
plt.ylabel('Rotation Angle (degree)')
plt.show()













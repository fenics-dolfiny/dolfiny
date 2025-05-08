import dolfinx.fem.function
import dolfinx.io.gmshio
from mpi4py import MPI

import dolfinx
import ufl
import dolfinx.fem.petsc
import numpy as np
import gmsh


# Initialise gmsh and set options
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

# Add model under given name
gmsh.model.add("")

origin = gmsh.model.occ.addPoint(0, 0, 0, 0.00001)
circle = gmsh.model.occ.addCircle(0, 0, 0, 1)
# circle = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
circleLoop = gmsh.model.occ.addCurveLoop([circle])
disk = gmsh.model.occ.addSurfaceFilling(circleLoop)
# Add the origin as a point (must use OCC if geometry is OCC-based)

# origin = gmsh.model.geo.addPoint(0, 0, 0, 0, tag=2)

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()
# gmsh.model.occ.fragment([(2, disk)], [(0, origin)])

embeded_origin = gmsh.model.mesh.embed(0, [origin], 2, disk)
gmsh.model.occ.fragment([(2, disk)], [(0, origin)])

embeded_origin = gmsh.model.mesh.getEmbedded(2, disk)[0][1]
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(0, [origin], tag=0)
gmsh.model.addPhysicalGroup(1, [circle], tag=1)
gmsh.model.addPhysicalGroup(2, [disk], tag=2)
# gmsh.model.addPhysicalGroup(0, [origin], tag=2)

# Generate the mesh
gmsh.model.mesh.generate(dim=2)
gmsh.write("circle_with_origin_fixed.msh")

mesh_data = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, 2)

mesh = mesh_data.mesh
print(mesh_data.cell_tags)
# exit()
# mesh = dolfinx.mesh.create_unit_square(
#     MPI.COMM_WORLD, 10, 10, cell_type=dolfinx.mesh.CellType.quadrilateral
# )
# mesh.topology.create_connectivity(0, 2)

source_vertex = dolfinx.mesh.locate_entities(
    mesh, 0, lambda x: np.isclose(x[0], 0.0, rtol=1e-3) & np.isclose(x[1], 0.0, rtol=1e-3)
)
print(source_vertex)
assert len(source_vertex) == 1
V = dolfinx.fem.functionspace(mesh, ("P", 5))
source_dof = dolfinx.fem.locate_dofs_topological(V, 0, source_vertex)
print(source_dof)
f = dolfinx.fem.Function(V)
f.x.array[source_dof] = 1
dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags.find(2))
print(mesh_data.cell_tags.find(2))
# exit()
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

dp = ufl.Measure("dx", domain=mesh, subdomain_data=[source_vertex], metadata={"quadrature_type": "vertex"})
L = v * f * dp

boundary_facets = mesh_data.facet_tags.find(1)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, 1, boundary_facets)
bcs = dolfinx.fem.dirichletbc(0.0, boundary_dofs, V)

prob = dolfinx.fem.petsc.LinearProblem(a, L, [bcs])

u = prob.solve()

u_ana = dolfinx.fem.Function(V, name="analytical")
x = ufl.SpatialCoordinate(mesh)
u_ana.interpolate(dolfinx.fem.Expression(-1/(2*np.pi)*ufl.ln(ufl.sqrt(x[0]**2 + x[1]**2)), V.element.interpolation_points))
# with dolfinx.io.XDMFFile(mesh.comm, "example.xdmf", "w") as file:
with dolfinx.io.VTXWriter(mesh.comm, "example.bp", [u]) as file:
    # file.write_mesh(mesh)
    # file.write_function(u)
    # file.write_function(u_ana)
    file.write(0.0)

# %% [markdown]
# # Obstacle problem
# This demo showcases how to solve obstacle problems with PETSc TAO's optimziation algorithms
# with the Dolfiny 🐬 interface.
# Both linear and non-linear formulations are demonstrated.

# ## Obstacle problems
# Let us take a surface $\mathcal{F} \subset \mathbb{R}^3$ which is parametrized by a regular
# paramatetrization $\Psi$ over $B \subset \mathbb{R}^2$, i.e. $\mathcal{F} = \Psi(B)$.
# The *suface area* of $\mathcal{F}$ is then given by
# $$
#   \int_\mathcal{F} 1 \, \text{d} x
#   = \int_{\Psi(B)} 1 \, \text{d} x
#   = \int_{B} | \det ( D\Psi(x) ) | \, \text{d} x
#   = \int_{B} \left| \frac{\partial \Psi}{\partial x} \times \frac{\partial \Psi}{\partial y} \right| \, \text{d} x.
# $$
# Let us now choose the coordinates $\Psi_1 = (x, y, g(x, y))$ and $\Psi_2 = (x, y, )$ TODO: continue
#
# ## Minimizing Dirichlet Energy
# TODO: another option where obstacle problems show up naturely
# %%

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import ufl
from dolfinx import fem, io, mesh

import numpy as np

# import dolfiny

n = 64
msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)

V = fem.functionspace(msh, ("Lagrange", 1))

boundary_dofs = fem.locate_dofs_topological(
    V,
    msh.topology.dim - 1,
    mesh.locate_entities_boundary(
        msh,
        dim=(msh.topology.dim - 1),
        marker=lambda x: np.isclose(x[0], 0.0)
        | np.isclose(x[0], 1.0)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], 1.0),
    ),
)
bc = fem.dirichletbc(value=0.0, dofs=boundary_dofs, V=V)


u = fem.Function(V, name="u")

# linearised surface area: (Dirichlet energy)
F = ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx
# non-linear surface area:
# F = ufl.sqrt(1 + ufl.inner(ufl.grad(u), ufl.grad(u))) * ufl.dx

# JF = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
JF = ufl.derivative(F, u, ufl.TestFunction(V))
HF = ufl.derivative(JF, u, ufl.TrialFunction(V))

F = fem.form(F)
JF = fem.form(JF)
HF = fem.form(HF)


def assemble_F(tao, x) -> float:
    # x.copy(u.x.petsc_vec)
    # x.assemble()
    # TODO: copying here directyl the petsc level data structures does not work, why?
    u.x.array[:] = x.getArray(readonly=True)[:]  # = x.copy()
    # print(x == u.x.petsc_vec)

    print(f"norm u: {u.x.petsc_vec.norm()}")
    # size_local = u[i].x.petsc_vec.getLocalSize()
    # u[i].x.petsc_vec.array[:] = x.array_r[offset : offset + size_local]
    # offset += size_local
    # u[i].x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
    #                                 mode=PETSc.ScatterMode.FORWARD)

    # dolfiny.function.vec_to_functions(x, [u])
    return fem.assemble_scalar(F)


def assemble_JF(tao, x, J):
    # x.copy(u.x.petsc_vec)
    u.x.array[:] = x.getArray(readonly=True)[:]

    with J.localForm() as J_local:
        J_local.set(0.0)

    dolfinx.fem.petsc.assemble_vector(J, JF)
    J.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # TODO: in the case that initial guess is not fullfilling boundary conditions we should correct here.
    # dolfinx.fem.petsc.apply_lifting(J, JF, bcs=[bc], x0=x, alpha=0.0)
    # dolfinx.fem.petsc.set_bc(J, [bc], alpha=0.0)
    # bc.set(J, [bc], alpha=0.0)
    print(f"norm J: {J.norm()}")
    # J.assemble()


def assemble_HF(tao, x, H, P):
    u.x.array[:] = x.getArray(readonly=True)[:]

    H.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(H, HF, [bc])
    H.assemble()

    print(f"Hessian norm: {H.norm()}")

    if P != H:
        P.assign(H)


# ineq constraint u >= h
x = ufl.SpatialCoordinate(msh)
h = ufl.conditional((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 <= 0.2**2, 1, 0)
interpolation_h = fem.Function(V, name="h")
interpolation_h.interpolate(fem.Expression(h, V.element.interpolation_points))

Jh = PETSc.Mat().createAIJ([u.x.array.size, u.x.array.size])
Jh.setUp()
for i in range(u.x.array.size):
    Jh.setValue(i, i, 1.0)
Jh.assemblyBegin()  # Assembling the matrix makes it "useable".
Jh.assemblyEnd()
# Jh.transpose()


# print(u.x.petsc_vec.getLocalSize())
# print(u.x.petsc_vec.getSize())
# print(Jh.getSize())
# print(Jh.getLocalSize())
def assemble_h(tao, x, c):
    # x.copy(c)
    # u.x.array[:] = x.getArray(readonly=True)[:]

    with c.localForm() as c_local:
        c_local.set(0.0)

    c.setArray(x.getArray(readonly=True))
    c.aypx(-1, interpolation_h.x.petsc_vec)
    # c.assemble()


def assemble_Jh(tao, x, J, P):
    # print(f"assemble_Jh J size: {J.getSize()} (row x col)")
    # print(f"assemble_Jh P size: {P.getSize()} (row x col")
    # pass
    if J != P:
        P.assign(J)
    print(f"Jh: {J.norm()}")


opts = PETSc.Options()
# opts.setValue("tao_type", "almm")
opts.setValue("tao_type", "almm")
opts.setValue("tao_ls_monitor", "")
opts.setValue("tao_max_it", "100")
# opts.setValue("tao_almm_type", "classic")
# opts.setValue("tao_fd_gradient", "")
# opts.setValue("on_error_attach_debugger", "")
# opts.setValue("tao_gatol", "0")
# opts.setValue("tao_grtol", "0")

tao = PETSc.TAO().create(MPI.COMM_WORLD)
tao.setFromOptions()
tao.setObjective(assemble_F)
tao.setGradient(assemble_JF, u.x.petsc_vec.copy())
# c = PETSc.Vec().create(tao.getComm())
# c = PETSc.Vec().createSeq(u.x.array.size)
c = u.x.petsc_vec.copy()
# c = dolfinx.la.petsc.create_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
# c.setSizes(1)
# c.setType("standard")
tao.setInequalityConstraints(assemble_h, c)
# tao.setEqualityConstraints(assemble_h, c) # u.x.petsc_vec.copy()

# J = PETSc.Mat().create(tao.getComm())
# J.setSizes([1, 2])
# J.setType(PETSc.Mat.Type.DENSE)
# J.setUp()

# tao.setGradient(Gradient(), None)
# Jh.mult(u.x.petsc_vec, )
tao.setJacobianInequality(assemble_Jh, Jh)

ub = fem.Function(V, name="upper_bound")
ub.x.array[:] = 1e4
bc.set(ub.x.array, alpha=1)

lb = fem.Function(V, name="lower_bound")
lb.x.array[:] = 0
bc.set(lb.x.array, alpha=1)


tao.setVariableBounds(lb.x.petsc_vec, ub.x.petsc_vec)
# tao.setJacobianEquality(assemble_Jh, Jh)
u.x.array[:] = 1
bc.set(u.x.array)
tao.setSolution(u.x.petsc_vec)
# tao.setType(PETSc.TAO.Type.ALMM)
tao.setTolerances(gatol=1.0e-4)
tao.solve()
# c.destroy()
tao.view()

with io.XDMFFile(msh.comm, "out_obstacle/data.xdmf", "w") as file:
    file.write_mesh(msh)

    file.write_function(interpolation_h)
    file.write_function(u)

# %%

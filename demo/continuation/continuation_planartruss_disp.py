#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import mesh_planartruss_gmshapi as mg
import numpy as np

import dolfiny

# Basic settings
name = "continuation_planartruss_disp"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
L = 1.0  # member length
θ = np.pi / 20  # angle
p = 1  # physics: polynomial order
q = 1  # geometry: polynomial order

# Create the regular mesh of a curve with given dimensions
gmsh_model, tdim = mg.mesh_planartruss_gmshapi(name, L=L, nL=2, θ=θ, order=q)

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim, prune_z=True)

# Get merged MeshTags for each codimension
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 0)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mesh, mts, tdim - 1)

# Define shorthands for labelled tags
support = interfaces_keys["support"]
connect = interfaces_keys["connect"]
verytop = interfaces_keys["verytop"]
upper = subdomains_keys["upper"]
lower = subdomains_keys["lower"]

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)
dS = ufl.Measure("dS", domain=mesh, subdomain_data=interfaces)

# Define elements
Ue = basix.ufl.element("P", mesh.basix_cell(), degree=p, shape=(2,))
Re = basix.ufl.element("P", mesh.basix_cell(), degree=p, shape=(2,))
Ke = basix.ufl.element("DP", mesh.basix_cell(), degree=0)
Se = basix.ufl.element("DP", mesh.basix_cell(), degree=p, shape=(2,))

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)
Rf = dolfinx.fem.functionspace(mesh, Re)
Kf = dolfinx.fem.functionspace(mesh, Ke)
Sf = dolfinx.fem.functionspace(mesh, Se)

# Define functions
u = dolfinx.fem.Function(Uf, name="u")  # displacement
r = dolfinx.fem.Function(Rf, name="r")  # constraint, Lagrange multiplier
k = dolfinx.fem.Function(Kf, name="k")  # axial stiffness
s = dolfinx.fem.Function(Sf, name="s")  # internal force
u_ = dolfinx.fem.Function(Uf, name="u_")  # displacement, inhomogeneous (bc)
δu = ufl.TestFunction(Uf)
δr = ufl.TestFunction(Rf)

# Define state as (ordered) list of functions
m, δm = [u, r], [δu, δr]

# System properties
k.x.array[dolfiny.mesh.locate_dofs_topological(Kf, subdomains, lower)] = (
    1.0e2  # axial stiffness, lower
)
k.x.array[dolfiny.mesh.locate_dofs_topological(Kf, subdomains, upper)] = (
    2.0e-0  # axial stiffness, upper
)
k.x.scatter_forward()

d = dolfinx.fem.Constant(mesh, [0.0, -1.0])  # disp vector, 2D

# Identify dofs of function spaces associated with tagged interfaces/boundaries
support_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, support)

# Set up restriction
rdofsU = dolfiny.mesh.locate_dofs_topological(Uf, subdomains, [lower, upper], unroll=True)
rdofsR = dolfiny.mesh.locate_dofs_topological(Rf, interfaces, verytop, unroll=True)
restrc = dolfiny.restriction.Restriction([Uf, Rf], [rdofsU, rdofsR])

# Define boundary conditions
bcs = [dolfinx.fem.dirichletbc(u_, support_dofs_Uf)]  # fix full displacement at support

# Tangent basis (un-deformed configuration)
t0 = ufl.geometry.Jacobian(mesh)[:, 0]
# Unit tangent basis
t0 /= ufl.sqrt(ufl.dot(t0, t0))
# Projector to tangent space
P = ufl.outer(t0, t0)

# Various expressions
I = ufl.Identity(2)  # noqa: E741
F = I + ufl.grad(u)

# Strain state (axial): from axial stretch λm
λm = ufl.sqrt(ufl.dot(F * t0, F * t0))  # deformed tangent t = F * t0
Em = P * (λm**2 - 1) / 2 * P  # Green-Lagrange strain
# Em = P * (λm - 1) * P  # Biot strain

# Virtual membrane strain
dEm = dolfiny.expression.derivative(Em, m, δm)

# Membrane stress
Sm = k * Em

# Load factor
λ = dolfinx.fem.Constant(mesh, scalar(1.0))

# Constraint
c = ufl.inner(r, λ * d - u)

# Weak form
form = -ufl.inner(dEm, Sm) * dx + dolfiny.expression.derivative(c, m, δm) * ds(verytop)

# Overall form (as list of forms)
forms = dolfiny.function.extract_blocks(form, δm)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_meshtags(mesh, mts) if q <= 2 else None

# Options for PETSc backend
opts = PETSc.Options("continuation")  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-09
opts["snes_rtol"] = 1.0e-09
opts["snes_stol"] = 1.0e-09
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"

monitor_history: list[np.ndarray] = []


def monitor(context=None):
    # obtain dual quantities for monitoring
    dolfiny.interpolation.interpolate(Sm * t0, s)  # internal force

    track = [
        (u, dolfiny.mesh.locate_dofs_topological(Uf, interfaces, verytop, unroll=True)),
        (u, dolfiny.mesh.locate_dofs_topological(Uf, interfaces, connect, unroll=True)),
        (s, dolfiny.mesh.locate_dofs_geometrical(Sf, interfaces, verytop, unroll=True)),
    ]

    values = []

    for function, dof_idx in track:
        bs = function.function_space.dofmap.bs
        ls = function.function_space.dofmap.index_map.size_local
        local_dof_idx = dof_idx[np.argwhere(dof_idx < ls * bs).squeeze()]
        values_owner = np.argmax(comm.allgather(local_dof_idx.size > 0))
        if comm.rank == values_owner:
            values_bcast = function.x.array[local_dof_idx].squeeze()
        else:
            values_bcast = None
        values.append(comm.bcast(values_bcast, root=values_owner))

    monitor_history.append(values)


def block_inner(a1, a2):
    b1, b2 = [], []
    for mi in m:
        b1.append(dolfinx.fem.Function(mi.function_space, name=mi.name))
        b2.append(dolfinx.fem.Function(mi.function_space, name=mi.name))
    restrc.vec_to_functions(a1, b1)  # restriction handles transfer
    restrc.vec_to_functions(a2, b2)
    inner = 0.0
    for b1i, b2i in zip(b1, b2):
        inner += dolfiny.expression.assemble(ufl.inner(b1i, b2i), ufl.dx(mesh))
    return inner


# Create nonlinear problem context
problem = dolfiny.snesblockproblem.SNESBlockProblem(
    forms, m, bcs, prefix="continuation", restriction=restrc
)

# Create continuation problem context
continuation = dolfiny.continuation.Crisfield(problem, λ, inner=block_inner)

# Initialise continuation problem
continuation.initialise(ds=0.02, λ=0.0)

# Monitor (initial state)
monitor()

# Continuation procedure
for j in range(35):
    dolfiny.utils.pprint(f"\n*** Continuation step {j:d}")

    # Solve one step of the non-linear continuation problem
    continuation.solve_step()

    # Monitor
    monitor()

    # Write output
    ofile.write_function(u, j) if q <= 2 else None

ofile.close()

# Post-processing
if comm.rank == 0:
    # plot
    import matplotlib.pyplot as plt
    from cycler import cycler

    flip = cycler(color=["tab:orange", "tab:blue"])
    flip += cycler(markeredgecolor=["tab:orange", "tab:blue"])
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=400)
    ax1.set_title(f"3-member planar truss, $u$-controlled, $θ$ = {θ / np.pi:1.2f}$π$", fontsize=12)
    ax1.set_xlabel("displacement $u$ $[m]$", fontsize=12)
    ax1.set_ylabel("internal force $N^{top}_1 (λ)$ $[kN]$", fontsize=12)
    ax1.grid(linewidth=0.25)
    # fig.tight_layout()

    # monitored results (force-displacement curves)
    um_ = np.array(monitor_history)[:, :2, 1]
    fm_ = np.array(monitor_history)[:, 2, 1]
    ax1.plot(um_, fm_, lw=1.5, ms=6.0, mfc="w", marker=".", label=["$u^{top}_1$", "$u^{mid}_1$"])

    ax1.legend()
    ax1.set_xlim((-0.4, +0.0))
    ax1.set_ylim((-0.2, +0.2))
    fig.savefig(f"{name}.png")

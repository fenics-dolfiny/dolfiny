#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import dolfinx.fem.petsc
import ufl
from dolfinx import default_scalar_type as scalar

import mesh_spatialtruss_gmshapi as mg
import numpy as np

import dolfiny

# Basic settings
name = "continuation_spatialtruss"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
L = 1.0  # member length
θ = np.pi / 3  # angle
p_physics = 1  # physics: polynomial order
p_geometry = 1  # geometry: polynomial order

# Create the regular mesh of a curve with given dimensions
gmsh_model, tdim = mg.mesh_spatialtruss_gmshapi(name, L=L, nL=2, θ=θ, order=p_geometry)

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim)
gdim = mesh.geometry.dim

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
Ue = basix.ufl.element("P", mesh.basix_cell(), degree=p_physics, shape=(gdim,))

# Define function spaces
Uf = dolfinx.fem.functionspace(mesh, Ue)

# Define functions
u = dolfinx.fem.Function(Uf, name="u")
u_ = dolfinx.fem.Function(Uf, name="u_")  # boundary conditions
δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf))
(δu,) = δm

# Define state as (ordered) list of functions
m = [u]

# System properties
K = dolfinx.fem.Constant(mesh, scalar(1.0))  # axial stiffness, lower
β = dolfinx.fem.Constant(mesh, scalar(1.2))  # stiffness factor for upper
p = dolfinx.fem.Constant(mesh, [0.0, 0.0, -1.0])  # load vector, 2D

# Identify dofs of function spaces associated with tagged interfaces/boundaries
support_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, support)
verytop_dofx_Uf = dolfiny.mesh.locate_dofs_topological(
    (Uf.sub(0), Uf.sub(0).collapse()[0]), interfaces, verytop
)
verytop_dofy_Uf = dolfiny.mesh.locate_dofs_topological(
    (Uf.sub(1), Uf.sub(1).collapse()[0]), interfaces, verytop
)

# Define boundary conditions
bcs = [
    dolfinx.fem.dirichletbc(u_, support_dofs_Uf),  # fix full displacement at support
    dolfinx.fem.dirichletbc(u_, verytop_dofx_Uf, Uf),  # fix x displacement at verytop
    dolfinx.fem.dirichletbc(u_, verytop_dofy_Uf, Uf),  # fix y displacement at verytop
]

# Tangent basis (un-deformed configuration)
t0 = ufl.geometry.Jacobian(mesh)[:, 0]
# Unit tangent basis
t0 /= ufl.sqrt(ufl.dot(t0, t0))
# Projector to tangent space
P = ufl.outer(t0, t0)

# Define EPS
EPS = dolfinx.fem.Constant(mesh, scalar(1.0e-10)) / ufl.JacobianDeterminant(mesh)

# Various expressions
I = ufl.Identity(mesh.geometry.dim)  # noqa: E741
F = I + ufl.grad(u)

# Strain state (axial): from axial stretch λ
λ = ufl.sqrt(ufl.dot(F * t0, F * t0) + EPS)  # deformed tangent t = F * t0
# Em = P * (λ**2 - 1) / 2 * P  # Green-Lagrange strain
Em = P * (λ - 1) * P  # Biot strain

# Virtual membrane strain
dEm = ufl.derivative(Em, m, δm)

# Membrane stress
Sm = K * Em

# load factor
λ = dolfinx.fem.Constant(mesh, scalar(0.1))

# Weak form
form = -ufl.inner(dEm, Sm) * dx(lower) - β * ufl.inner(dEm, Sm) * dx(upper)
form += λ * ufl.inner(δu, p) * ds(verytop)

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_meshtags(mesh, mts) if p_geometry <= 2 else None

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

u_step: list[np.ndarray] = []
λ_step: list[np.ndarray] = []


def monitor(context=None):
    if comm.size > 1:
        return

    u2_component = (Uf.sub(2), Uf.sub(2).collapse()[0])

    track_ids = [
        dolfiny.mesh.locate_dofs_topological(u2_component, interfaces, verytop),
        dolfiny.mesh.locate_dofs_topological(u2_component, interfaces, connect),
    ]

    track_val = [u.x.petsc_vec[idx[0]].squeeze() for idx in track_ids]

    u_step.append(track_val)
    λ_step.append(context.λ.value.item())


def block_inner(a1, a2):
    b1, b2 = [], []
    for mi in m:
        b1.append(dolfinx.fem.Function(mi.function_space, name=mi.name))
        b2.append(dolfinx.fem.Function(mi.function_space, name=mi.name))
    dolfinx.fem.petsc.assign(a1, b1)
    dolfinx.fem.petsc.assign(a2, b2)

    [b.x.scatter_forward() for b in b1 + b2]

    inner = 0.0
    for b1i, b2i in zip(b1, b2):
        inner += dolfiny.expression.assemble(ufl.inner(b1i, b2i), ufl.dx(mesh))
    return inner


# Create nonlinear problem context
problem = dolfiny.snesproblem.SNESProblem(forms, m, bcs, prefix="continuation")

# Create continuation problem context
continuation = dolfiny.continuation.Crisfield(problem, λ, inner=block_inner)

# Initialise continuation problem
continuation.initialise(ds=0.05, λ=0.0)

# Monitor (initial state)
monitor(continuation)

# Continuation procedure
for j in range(60):
    dolfiny.utils.pprint(f"\n*** Continuation step {j:d}")

    # Solve one step of the non-linear continuation problem
    continuation.solve_step(ds=0.06)

    # Monitor
    monitor(continuation)

    # Write output
    ofile.write_function(u, j) if p_geometry <= 2 else None

ofile.close()

# Post-processing
if comm.size == 1:
    # plot
    import matplotlib.pyplot as plt
    from cycler import cycler

    flip = cycler(color=["tab:orange", "tab:blue"])
    flip += cycler(markeredgecolor=["tab:orange", "tab:blue"])
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=400)
    ax1.set_title(f"4-member spatial truss, $θ$ = {θ / np.pi:1.3f}$π$", fontsize=12)
    ax1.set_xlabel("displacement $u / L$ $[-]$", fontsize=12)
    ax1.set_ylabel("load factor $λ$ $[-]$", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    # monitored results (load-response curves)
    ax1.plot(
        np.array(u_step) / L,
        λ_step,
        lw=1.5,
        ms=6.0,
        mfc="w",
        marker=".",
        label=["$u^{top}_2$", "$u^{mid}_2$"],
    )

    ax1.legend()
    ax1.set_xlim((-2.0, +0.0))
    ax1.set_ylim((-0.8, +0.8))
    fig.savefig(f"{name}.png")

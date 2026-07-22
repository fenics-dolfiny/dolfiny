"""
Systematic sweep of PETSc SNESNEWTONLS line-search strategies on the HL 2D
double-arch TMC problem (see scripts/double_arch_HL.py for the single-run reference).

Mesh, function spaces, kinematics, energies, BCs and the residual form are
built exactly once and reused across variants; 
only the SNES line-search configuration changes between runs. 
Each variant repeats the SAME fixed (non-adaptive) load-stepping schedule so the
comparison isolates the line-search effect only.

Outputs:
  - <label>.npz       loading_history / m_node_displ per variant (same schema
                       as double_arch_HL.py, for reuse with postprocessing.py)
  - results.csv        label, max_load, total_iters, wall_time_s, reason
  - sweep_summary.png  bar chart of max load factor reached per variant
"""

from mpi4py import MPI
import dolfinx
import ufl
import numpy as np
import csv
import time
from pathlib import Path
from double_arch_gmsh_NEW import mesh_double_arch_gmsh  # external-only full-box TM 
# from double_arch_gmsh import mesh_double_arch_gmsh

from dolfinx import fem
from dolfiny.utils import pprint
from petsc4py import PETSc

# Basic settings
name = "double_arch_HL"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters (identical to double_arch_HL.py)
Lx, H, Lz = 260., 50., 50.
Di, t = 90., 5.
lR = 10.
g0 = 20.

nL, nH, nt = 24, 10, 1
nDi = 12
nTM = 10

cell_tags = {"block": 1, "arch1": 2, "arch2": 3, "tm": 4}
facet_tags = {"bottom": 1, "top_arches": 2}

verbosity = 1
dim = 2
sweep_name = f"{name}_{dim}D_sweep"

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / sweep_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = mesh_double_arch_gmsh(
    tdim=dim, cell_tags=cell_tags, facet_tags=facet_tags, Lx=Lx, Lz=Lz, H=H, Di=Di, t=t, g0=g0, lR=lR,
    nL=nL, nH=nH, nt=nt, nDi=nDi, nTM=nTM,
    verbosity=verbosity, comm=comm, name=name)

mesh_data = dolfinx.io.gmsh.model_to_mesh(model, comm, rank=0, gdim=dim)
mesh = mesh_data.mesh
ct = mesh_data.cell_tags
ft = mesh_data.facet_tags

tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)

num_cells_owned = mesh.topology.index_map(tdim).size_local
num_nodes_owned = mesh.topology.index_map(0).size_local
num_cells_global = comm.allreduce(num_cells_owned, op=MPI.SUM)
num_nodes_global = comm.allreduce(num_nodes_owned, op=MPI.SUM)
pprint(f"Mesh: {num_cells_global} cells, {num_nodes_global} nodes")

metadata = {"quadrature_rule": "GLL", "quadrature_degree": 3}
dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct, metadata=metadata)
dxA1 = dx(cell_tags["arch1"])
dxA2 = dx(cell_tags["arch2"])
dxThird = dx(cell_tags["tm"])
dxBlock = dx(cell_tags["block"])
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

third_medium_mesh, medium_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(cell_tags["tm"])
)[0:2]

element_deg = 2
V = fem.functionspace(mesh, ("Lagrange", element_deg, (tdim,)))

u = fem.Function(V, name="displacement")
δu = ufl.TestFunction(V)

I = ufl.Identity(len(u))
F_2D = ufl.variable(I + ufl.grad(u))
C_2D = ufl.variable(F_2D.T * F_2D)
I1 = ufl.tr(C_2D) + 1
J = ufl.det(F_2D)

nu = 0.3
E_a1 = 1e5
E_a2 = 1e3
E_block = 300.

mu_a1 = fem.Constant(mesh, E_a1 / (2 * (1 + nu)))
lam_a1 = fem.Constant(mesh, E_a1 * nu / ((1 + nu) * (1 - 2 * nu)))
mu_a2 = fem.Constant(mesh, E_a2 / (2 * (1 + nu)))
lam_a2 = fem.Constant(mesh, E_a2 * nu / ((1 + nu) * (1 - 2 * nu)))
mu_block = fem.Constant(mesh, E_block / (2 * (1 + nu)))
lam_block = fem.Constant(mesh, E_block * nu / ((1 + nu) * (1 - 2 * nu)))

Psi_a1 = mu_a1 / 2 * (I1 - 3) - mu_a1 * ufl.ln(J) + lam_a1 / 2 * ufl.ln(J)**2
Psi_a2 = mu_a2 / 2 * (I1 - 3) - mu_a2 * ufl.ln(J) + lam_a2 / 2 * ufl.ln(J)**2
Psi_block = mu_block / 2 * (I1 - 3) - mu_block * ufl.ln(J) + lam_block / 2 * ufl.ln(J)**2

Pi = Psi_a1 * dxA1 + Psi_a2 * dxA2 + Psi_block * dxBlock

K_TM = E_block / (3 * (1 - 2 * nu))
mu_tm = mu_block
Psi_tm = mu_tm / 2 * (J ** (-2/3) * I1 - 3)  # 2D plane-strain: deviatoric term only

gamma = fem.Constant(mesh, 1.0e-4)
Pi_third = gamma * Psi_tm * dxThird

L_i = np.zeros(tdim)
for d in range(mesh.geometry.dim):
    x_i_max = mesh.comm.allreduce(mesh.geometry.x[:, d].max(), op=MPI.MAX)
    x_i_min = mesh.comm.allreduce(mesh.geometry.x[:, d].min(), op=MPI.MIN)
    L_i[d] = x_i_max - x_i_min
Ell = dolfinx.fem.Constant(mesh, np.max(L_i))
alpha = fem.Constant(mesh, 1.0e-04)
k_r = fem.Constant(mesh, alpha.value * Ell.value**2 * K_TM)

Hu = ufl.grad(ufl.grad(u))
Lu = ufl.div(ufl.grad(u))
HuHu = ufl.inner(Hu, Hu)
LuLu = ufl.inner(Lu, Lu) / ufl.tr(I)
Pi_HuLu = k_r / 2 * (HuHu - LuLu) * dxThird
Pi_R = Pi_HuLu

bottom_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(facet_tags["bottom"]))
bc_bottom = dolfinx.fem.dirichletbc(
    np.zeros(tdim, dtype=dolfinx.default_scalar_type), bottom_dofs, V)

top_arches_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(facet_tags["top_arches"]))
applied_disp = dolfinx.fem.Constant(mesh, (0.,) * tdim)
bc_top_arches = dolfinx.fem.dirichletbc(applied_disp, top_arches_dofs, V)

bcs = [bc_bottom, bc_top_arches]

residual = ufl.derivative(Pi + Pi_third + Pi_R, u, δu)

mesh.topology.create_connectivity(0, tdim)
m_node = dolfinx.mesh.locate_entities(mesh, 0, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], g0 + t))
dofs_point_y = dolfinx.fem.locate_dofs_topological(V.sub(1), 0, m_node)

# Fixed (non-adaptive) load schedule, identical to double_arch_HL.py with adaptive_load=False
full_disp = -90.0
pre_contact_disp_step = 1.0
contact_disp_step = 0.5
threshold = 20.0
dl_free = pre_contact_disp_step / abs(full_disp)
dl_contact = contact_disp_step / abs(full_disp)

# Line-search variants to sweep, with the base petsc_options shared by all
BASE_OPTIONS = {
    "snes_type": "newtonls",
    "snes_rtol": 1.0e-08,
    "snes_atol": 1.0e-08,
    "snes_max_it": 50,
    "snes_monitor": None,
    "snes_converged_reason": None,
    "ksp_type": "preonly",
    "pc_type": "cholesky",
    "pc_factor_mat_solver_type": "mumps",
}

VARIANTS = [
    ("basic", {"snes_linesearch_type": "basic"}),
    ("bt_order1", {"snes_linesearch_type": "bt", "snes_linesearch_order": 1}),
    ("bt_order2", {"snes_linesearch_type": "bt", "snes_linesearch_order": 2}),
    ("bt_order3", {"snes_linesearch_type": "bt", "snes_linesearch_order": 3}),
    ("l2", {"snes_linesearch_type": "l2"}),
    ("cp", {"snes_linesearch_type": "cp"}),
    ("nleqerr", {"snes_linesearch_type": "nleqerr"}),
]

results = []

for label, ls_options in VARIANTS:
    pprint("=" * 60)
    pprint(f"Variant: {label}  ({ls_options})")
    pprint("=" * 60)

    u.x.array[:] = 0.0
    u_prev = u.x.array.copy()

    petsc_options = dict(BASE_OPTIONS)
    petsc_options.update(ls_options)

    problem = dolfinx.fem.petsc.NonlinearProblem(
        residual,
        u,
        bcs=bcs,
        entity_maps=[medium_map],
        petsc_options_prefix=f"{sweep_name}_{label}_",
        petsc_options=petsc_options,
    )

    loading_history = np.empty(0, dtype=np.float64)
    m_node_displ = np.empty(0, dtype=np.float64)
    ii_load = 1

    total_iterations = 0
    ii = 1
    load = dl_free
    dl = dl_free
    reason = None
    max_load_reached = 0.0
    t_start = time.time()

    while load <= (1.0 + 1e-6):
        applied_disp.value[1] = full_disp * load
        pprint(f"\n Load step {ii}: u_y = {applied_disp.value[1]:.3f} (load={load:.4f})", flush=True)

        problem.solve()
        reason = problem.solver.getConvergedReason()
        total_iterations += problem.solver.getIterationNumber()

        if reason < 0:
            pprint(f"  Variant {label}: solver failed at load={load:.4f}, reason={reason}. Stopping variant.")
            u.x.array[:] = u_prev.copy()
            break

        max_load_reached = load
        current_disp = abs(full_disp) * load
        dl = dl_free if current_disp < threshold else min(dl, dl_contact)

        if current_disp >= g0:
            loading_history = np.append(loading_history, ii_load)
            m_node_displ = np.append(m_node_displ, np.asarray(u.x.array[dofs_point_y]).reshape(-1)[0])
            ii_load += 1

        u_prev[:] = u.x.array.copy()
        load += dl
        ii += 1

    wall_time_s = time.time() - t_start

    np.savez(OUT_DIR / f"{label}.npz", loading_history=loading_history, m_node_displ=m_node_displ)

    results.append({
        "label": label,
        "max_load": max_load_reached,
        "total_iters": total_iterations,
        "wall_time_s": wall_time_s,
        "reason": int(reason) if reason is not None else None,
    })

    pprint(f"Variant {label}: max_load={max_load_reached:.4f}, "
           f"total_iters={total_iterations}, wall_time={wall_time_s:.1f}s, reason={reason}")

# Write results.csv
results_csv = OUT_DIR / "results.csv"
with open(results_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["label", "max_load", "total_iters", "wall_time_s", "reason"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

pprint(f"\nResults written to {results_csv}")

# Summary bar chart: max load factor reached per variant
import matplotlib.pyplot as plt

labels = [r["label"] for r in results]
max_loads = [r["max_load"] for r in results]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, max_loads, color="#4C72B0")
ax.axhline(1.0, color="#55A868", linestyle="--", linewidth=1.5, label="full load (target)")
ax.set_ylabel("Max load factor reached", fontsize=12)
ax.set_xlabel("SNES line-search type", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, max_loads):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}",
            ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "sweep_summary.png", dpi=300)
plt.close()

pprint(f"Summary plot written to {OUT_DIR / 'sweep_summary.png'}")

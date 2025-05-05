from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

import dolfinx
import dolfinx.fem.petsc
import dolfinx.la.petsc
import ufl

import numpy as np

import dolfiny
import dolfiny.taoblockproblem

# ASTM A-36 steel as material
material = {
    "young_modulus": 2.11e11,  # Pa
    "poisson_ratio": 0.29,  # dimensionless
    "density": 7874,  # kg/m^3
    "yield_stress": 2.2e8,  # Pa
}

delta = 1e-2
alpha = 1e-4
eps = 1e-2


def double_well(v):
    return 9 / 16 * (v**2 - 1) ** 2


comm = MPI.COMM_WORLD

#   domain/mesh
lower = np.array([0, 0])
upper = np.array([1, 1])
elements_per_unit = 100
domain = dolfinx.mesh.create_rectangle(
    comm,
    [lower, upper],
    (upper - lower) * elements_per_unit,
    cell_type=dolfinx.mesh.CellType.triangle,
)

V_vec = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (2,)))
V_scalar = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

dof_lhs = dolfinx.mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 0.0))
dbc_u = dolfinx.fem.dirichletbc(
    np.zeros(2, dtype=ScalarType),
    dolfinx.fem.locate_dofs_topological(V_vec, entity_dim=1, entities=dof_lhs),
    V_vec,
)


def on_rhs(x):
    return np.isclose(x[0], 1.0) & np.greater_equal(x[1], 0.45) & np.less_equal(x[1], 0.55)


dof_rhs = dolfinx.mesh.locate_entities_boundary(domain, 1, on_rhs)
dbc_v = dolfinx.fem.dirichletbc(
    ScalarType(1),
    dolfinx.fem.locate_dofs_topological(V_scalar, entity_dim=1, entities=dof_rhs),
    V_scalar,
)

x = ufl.SpatialCoordinate(domain)
f = ufl.as_vector((0, -8.6e4))  # Nw
dS_facets = np.sort(dolfinx.mesh.locate_entities_boundary(domain, 1, on_rhs))
facet_tag = dolfinx.mesh.meshtags(domain, 1, dS_facets, np.full_like(dS_facets, 1))
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
# dS = ufl.Measure("dS", mesh, subdomain_data=)


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    nu = material["poisson_ratio"]
    E = material["young_modulus"]
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


def chi(v):
    return 1 / 4 * (v + 1) ** 2


def E(u, v):
    E = ((1 - delta) * chi(v) + delta) * 1 / 2 * ufl.inner(sigma(u), epsilon(u)) * ufl.dx

    compliance = ufl.inner(f, u) * ds(1)
    E -= compliance

    return E


v = dolfinx.fem.Function(V_scalar, name="v")  # phase-field (control)
u = dolfinx.fem.Function(V_vec, name="u")  # displacement (state)
p = dolfinx.fem.Function(V_vec, name="p")  # adjoint state


# fem.petsc.NonlinearProblem(ufl.derivative(E(u, v), u))
# duE = ufl.derivative(E(u, v), u, ufl.TestFunction(V_vec))

a = (
    ((1 - delta) * chi(v) + delta)
    * ufl.inner(sigma(ufl.TrialFunction(V_vec)), epsilon(ufl.TestFunction(V_vec)))
    * ufl.dx
)
L = ufl.inner(f, ufl.TestFunction(V_vec)) * ds(1)

elas_prob = dolfinx.fem.petsc.LinearProblem(
    a, L, [dbc_u], u, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
elas_prob.solve()


def J(u, v):
    P = (
        alpha
        * 0.5
        * (eps * ufl.inner(ufl.grad(v), ufl.grad(v)) + 1 / eps * double_well(v))
        * ufl.dx
    )  # + beta * chi(v) * ufl.dx
    J = P
    J += ufl.inner(f, u) * ds(1)

    return J


def C(u):
    return ufl.inner(f, u) * ds(1)


def P(v):
    return (
        alpha / 2 * (eps * ufl.inner(ufl.grad(v), ufl.grad(v)) + 1 / eps * double_well(v)) * ufl.dx
    )  # + beta * chi(v) * ufl.dx


p = dolfinx.fem.Function(V_vec)


@dolfiny.taoblockproblem.link_state(v)
def Jhat(tao, v_vec):
    # Taken care of in the
    # dbc_v.set(v.x.array)

    elas_prob.solve()

    # p = -u
    u.x.petsc_vec.copy(p.x.petsc_vec)
    p.x.petsc_vec.scale(-1)
    p.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    val = dolfinx.fem.assemble_scalar(J)
    val = comm.allreduce(val, op=MPI.SUM)
    # print(f'J(v) = {val}', flush=True)
    return val


J = J(u, v)
dv_DJ_hat = dolfinx.fem.form(
    ufl.derivative(ufl.action(ufl.derivative(E(u, v), u), p), v) + ufl.derivative(J, v)
)

J = dolfinx.fem.form(J)


@dolfiny.taoblockproblem.link_state(v)
def DJhat(tao, v_vec, G):
    # TODO: surely not necessary?
    # elas_prob.solve()

    # # p = -u
    # u.x.petsc_vec.copy(p.x.petsc_vec)
    # p.x.petsc_vec.scale(-1)

    with G.localForm() as g_local:
        g_local.set(0.0)

    dolfinx.fem.petsc.assemble_vector(G, dv_DJ_hat)
    G.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


g = [(v + 1) / 2 * ufl.dx == 0.3]
Jg = [[ufl.TestFunction(V_scalar) / 2 * ufl.dx]]

options = PETSc.Options()
options["tao_type"] = "almm"
options["tao_gatol"] = 1e-6
options["tao_gttol"] = 1e-6
options["tao_grtol"] = 1e-6
options["tao_ctol"] = 1e-3
options["tao_almm_subsolver_tao_type"] = "bqnls"
options["tao_almm_subsolver_ksp_type"] = "preonly"
options["tao_almm_subsolver_pc_type"] = "lu"
options["tao_almm_subsolver_pc_factor_mat_solver_type"] = "mumps"
options["tao_almm_subsolver_tao_monitor"] = ""
options["tao_monitor"] = ""

tao = dolfiny.taoblockproblem.TAOBlockProblem(
    Jhat, [v], J=(DJhat, v.x.petsc_vec.copy()), g=g, Jg=Jg, prefix=""
)

lb = dolfinx.fem.Function(V_scalar)
lb.x.petsc_vec.set(-1)

ub = dolfinx.fem.Function(V_scalar)
ub.x.petsc_vec.set(1)

tao.tao.setVariableBounds(lb.x.petsc_vec, ub.x.petsc_vec)
tao.tao.setMaximumIterations(100)
tao.solve([v])


with dolfinx.io.XDMFFile(domain.comm, "shop_vc_u.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(u)

with dolfinx.io.XDMFFile(domain.comm, "shop_vc_v.xdmf", "w") as file:
    file.write_mesh(domain)
    file.write_function(v)

# %% [markdown]
# # Dimensional Analysis of the Nernst-Planck Equations

# %%

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import sympy.physics.units as syu

import dolfiny

# import dolfiny.ufl_utils
from dolfiny.units import Quantity

# references:
# https://doi.org/10.1524/zpch.1954.1.5_6.305
# https://doi.org/10.1006/jcis.1999.6145

# Basic settings
name = "pnp_4species_steadystate"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
xp = [0.0, 1e7]
ne = 25  # number of elements

# Create mesh
mesh = dolfinx.mesh.create_interval(comm, ne, xp)  # 1d
# mesh = dolfinx.mesh.create_rectangle(comm, [[xp[0], -0.5], [xp[1], +0.5]], [ne, 8])  # 2d
# mesh = dolfinx.mesh.create_box(comm, [[0, -0.5, -0.5], [xp[1], +0.5, +0.5]], [ne, 8, 8])  # 3d

# Create mesh tags
subdomain_tdim = mesh.topology.dim
interface_tdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(interface_tdim, subdomain_tdim)
mesh.topology.create_connectivity(subdomain_tdim, subdomain_tdim)
facets_1 = dolfinx.mesh.locate_entities(mesh, interface_tdim, lambda x: np.isclose(x[0], xp[0]))
facets_2 = dolfinx.mesh.locate_entities(mesh, interface_tdim, lambda x: np.isclose(x[0], xp[1]))
cells_iv = dolfinx.mesh.locate_entities(
    mesh, subdomain_tdim, lambda x: (x[0] >= xp[0]) & (x[0] <= xp[1])
)
mts = {
    "one": dolfinx.mesh.meshtags(mesh, interface_tdim, np.unique(facets_1), 0),  # tag = 0
    "two": dolfinx.mesh.meshtags(mesh, interface_tdim, np.unique(facets_2), 1),  # tag = 1
    "domain": dolfinx.mesh.meshtags(mesh, subdomain_tdim, np.unique(cells_iv), 0),  # tag = 0
}

# Merge meshtags, see `boundary_keys` for identifiers of outer faces
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mesh, mts, subdomain_tdim)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mesh, mts, interface_tdim)

# System constants (with units)
ε_r = Quantity(mesh, 80.0, sy.S(1), "ε_r")
ε_0 = Quantity(mesh, 8.854187819e-12, syu.farad / syu.meter, "ε_0")
e_0 = Quantity(mesh, 1.602176634e-19, syu.coulomb, "e_0")
F = Quantity(mesh, 9.648533212e04, syu.coulomb / syu.mol, "F")
R = Quantity(mesh, 8.314462618, syu.joule / syu.kelvin / syu.mol, "R")
T = Quantity(mesh, 300.0, syu.kelvin, "T")
c_ref = Quantity(mesh, 50.0, syu.mol / syu.meter**3, "c_ref")
φ_ref = Quantity(mesh, 1.0, syu.volt, "φ_ref")
l_ref = Quantity(mesh, 1, syu.angstrom, "l_ref")
D_ref = Quantity(mesh, 1.0e-10, syu.meter**2 / syu.second, "D_ref")

# %%

# System parameters
species = {
    "$ SO^{2-}_{4} $": dict(
        z=Quantity(mesh, -2, sy.S(1), "z_SO4"),  # valence number
        D=3 * D_ref,  # diffusivity
        a=6 * l_ref,  # ion radius
    ),
    "$ Mg^{2+} $": dict(
        z=Quantity(mesh, 2, sy.S(1), "z_Mg"),  # valence number
        D=3 * D_ref,  # diffusivity
        a=8 * l_ref,  # ion radius
    ),
    "$ Na^{+} $": dict(
        z=Quantity(mesh, 1, sy.S(1), "z_Na"),  # valence number
        D=3 * D_ref,  # diffusivity
        a=4 * l_ref,  # ion radius
    ),
    "$ K^{+} $": dict(
        z=Quantity(mesh, 1, sy.S(1), "z_K"),  # valence number
        D=5 * D_ref,  # diffusivity
        a=3 * l_ref,  # ion radius
    ),
}

z = ufl.as_vector([d["z"] for d in species.values()])  # [-] -- valence number
D = ufl.as_vector([d["D"] for d in species.values()])  # [m^2 / s] -- diffusivity
a = ufl.as_vector([d["a"] for d in species.values()])  # [m] -- ion radius
n = len(species.keys())  # number of species

c_1 = np.array([6.0, 3, 0, 0]) / 6  # [mmol / l] = [mol / m^3] -- concentrations left
c_2 = np.array([5.0, 0, 1, 3]) / 6  # [mmol / l] = [mol / m^3] -- concentrations right

φ_1 = scalar(0.0)  # [V] -- voltage left
φ_2 = R.value * T.value / F.value / φ_ref.value * 5  # [V] -- voltage right (RT / F * ln(ζ))

w = -ufl.dot(
    c_ref * ufl.as_vector(c_1), z
)  # [mol / m^3] -- fixed charge concentration (electroneutrality)

# %%

# Define elements
porder = 1  # ansatz order for physics
Ce = basix.ufl.element("P", mesh.basix_cell(), porder, shape=(n,))
Pe = basix.ufl.element("P", mesh.basix_cell(), porder)

# Define function spaces
Cf = dolfinx.fem.functionspace(mesh, Ce)
Pf = dolfinx.fem.functionspace(mesh, Pe)

# Define functions
c = dolfinx.fem.Function(Cf, name="c")  # [mol / m^3] -- concentration
φ = dolfinx.fem.Function(Pf, name="φ")  # [V] -- electrostatic potential

δc, δφ = ufl.TestFunctions(ufl.MixedFunctionSpace(Cf, Pf))

# Define state as (ordered) list of functions
m, δm = [c, φ], [δc, δφ]

# Create other functions: output / visualisation
vorder = mesh.geometry.cmap.degree
co = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (n,))), name="c")
φo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder)), name="φ")

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)

# %%

terms_φ = {
    "potential": ufl.inner(ufl.grad(δφ), ε_r * ε_0 * ufl.grad(φ)) * dx,
    "electroneutrality": -δφ * F * (ufl.dot(z, c) + w) * dx,
}
terms_c = {
    "diffusion": ufl.form.Zero(),
    "convection": ufl.form.Zero(),
    "debye_0th": ufl.form.Zero(),
    "debye_1st": ufl.form.Zero(),
}

quantities = [c_ref, φ_ref, D_ref, ε_0, F, R, T, l_ref, e_0]

if comm.rank == 0:
    dolfiny.units.buckingham_pi_analysis(quantities)

# %%
mapping = {
    mesh.ufl_domain(): l_ref,
    c: c_ref * c,
    φ: φ_ref * φ,
    δc: c_ref * δc,
    δφ: φ_ref * δφ,
}

# Expressions, (extended) Debye-Hückel model
ε = ε_0 * ε_r
A = ufl.sqrt(2) * F**2 * e_0 / (8 * ufl.pi) / ufl.sqrt((ε * R * T) ** 3)  # parameter A
B = ufl.sqrt(2) * F / ufl.sqrt(ε_0 * ε_r * R * T)  # parameter B
S = ufl.dot(ufl.elem_mult(z, z), c) / 2 + w * n / 2  # ionic strength [mol / m^3]

dimsys = syu.si.SI.get_dimension_system()
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(S, quantities, mapping), syu.amount_of_substance / syu.length**3
)
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(ε, quantities, mapping), syu.charge / (syu.voltage * syu.length)
)
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(B, quantities, mapping),
    syu.length ** (-1)
    * syu.amount_of_substance ** sy.Rational(-1, 2)
    * syu.length ** sy.Rational(3, 2),
)
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(A, quantities, mapping),
    syu.amount_of_substance ** sy.Rational(-1, 2) * syu.length ** sy.Rational(3, 2),
)

for δc_k, c_k, z_k, D_k, a_k in zip(δc, c, z, D, a):  # add species
    # Expansion of the Debye-Hueckel model
    # 0-th order term
    deb_0 = -A * ufl.sqrt(S) * z_k**2

    # 1-st order
    deb_1 = A * S * z_k**2 * B * a_k

    # advective velocity [1 / m]
    vk = z_k * F / (R * T) * ufl.grad(φ)

    # ion flux = (isotropic) diffusion + electro-migration
    fk_fick = -D_k * ufl.grad(c_k)
    fk_conv = -D_k * vk * c_k

    terms_c["diffusion"] -= ufl.inner(ufl.grad(δc_k), fk_fick) * dx
    terms_c["convection"] -= ufl.inner(ufl.grad(δc_k), fk_conv) * dx

    fk_deb_0 = -D_k * ufl.grad(deb_0) * c_k
    fk_deb_1 = -D_k * ufl.grad(deb_1) * c_k

    terms_c["debye_0th"] -= ufl.inner(ufl.grad(δc_k), fk_deb_0) * dx
    terms_c["debye_1st"] -= ufl.inner(ufl.grad(δc_k), fk_deb_1) * dx


# Few dimensional sanity checks
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(ufl.dot(z, c) - w, quantities, mapping),
    syu.amount_of_substance / syu.length**3,
)

assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(terms_φ["potential"], quantities, mapping),
    dolfiny.units.get_dimension(terms_φ["electroneutrality"], quantities, mapping),
)

# %%

factorized_φ = dolfiny.units.factorize(terms_φ, quantities, mode="factorize", mapping=mapping)
assert isinstance(factorized_φ, dict)
normalized_φ = dolfiny.units.normalize(factorized_φ, "potential", quantities)

factorized_c = dolfiny.units.factorize(terms_c, quantities, mode="factorize", mapping=mapping)
assert isinstance(factorized_c, dict)
normalized_c = dolfiny.units.normalize(factorized_c, "diffusion", quantities)

form = sum(list(normalized_φ.values()) + list(normalized_c.values()), ufl.form.Zero())

assert isinstance(form, ufl.form.Form)
# Check that the form is dimensionally consistent
form_dim = dolfiny.units.get_dimension(form, quantities)

# %%

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
bcsdofs_Cf_one = dolfiny.mesh.locate_dofs_topological(Cf, interfaces, interfaces_keys["one"])
bcsdofs_Cf_two = dolfiny.mesh.locate_dofs_topological(Cf, interfaces, interfaces_keys["two"])
bcsdofs_Pf_one = dolfiny.mesh.locate_dofs_topological(Pf, interfaces, interfaces_keys["one"])
bcsdofs_Pf_two = dolfiny.mesh.locate_dofs_topological(Pf, interfaces, interfaces_keys["two"])

# Boundary conditions
bcs = [
    # concentrations
    dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, c_1), bcsdofs_Cf_one, Cf),
    dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, c_2), bcsdofs_Cf_two, Cf),
    # potential
    dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, φ_1), bcsdofs_Pf_one, Pf),
    dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, φ_2), bcsdofs_Pf_two, Pf),
]

# Options for PETSc backend
opts = PETSc.Options(name)  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "none"
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-09
opts["snes_stol"] = 1.0e-06
opts["snes_max_it"] = 10
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(forms, m, bcs, prefix=name)

# Solve nonlinear problem
problem.solve()

# Assert convergence of nonlinear solver
problem.status(verbose=True, error_on_failure=True)

# Interpolate for output
dolfiny.interpolation.interpolate(c, co)
dolfiny.interpolation.interpolate(φ, φo)

# Write to xdmf file -- open in Paraview with Xdmf3ReaderT
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh)
    ofile.write_function(co)
    ofile.write_function(φo)

# Evaluate solution on path
n_ = ne + 1
x_ = np.vstack([np.linspace(xp[0], xp[1], n_), np.zeros((n_,)), np.zeros((n_,))])

c_ = dolfiny.function.evaluate(c, x_)
φ_ = dolfiny.function.evaluate(φ, x_)

# Plot (only on rank = 0)
if comm.rank == 0:
    title = "Ion transport ($Mg SO_4$ | $Na_2 SO_4 + K_2 SO_4$)"

    fig, ax1 = plt.subplots(dpi=400)
    ax1.set_title(f"{title}, steady-state", fontsize=12)
    ax1.set_xlabel("coordinate $x$ $[m]$", fontsize=12)
    ax1.set_ylabel("concentration $c_k$ $[mol/m^3]$", fontsize=12)
    ax1.grid(linewidth=0.25)
    ax2 = ax1.twinx()
    ax2.set_ylabel("electrostatic potential $φ$ $[V]$", fontsize=12)

    ax1.plot(l_ref.value * x_[0], c_ref.value * c_, "-", label=species.keys())
    ax1.ticklabel_format(style="sci", scilimits=(0, -2), axis="x")
    ax1.set_xlim((l_ref.value * xp[0], l_ref.value * xp[1]))
    ax1.set_ylim((0, c_ref.value))
    ax1.legend(loc="best")

    ax2.plot(l_ref.value * x_[0], φ_ref.value * φ_, "-", color="black", alpha=0.5)
    ax2.set_yticks([0, 0.03, 0.06, 0.09, 0.12, 0.15])
    ax2.set_ylim((0, 0.15))

    fig.tight_layout()
    fig.savefig(f"{name}.pdf")

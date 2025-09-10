# %% [markdown]
# # Hyperelasticity in spectral formulation

# This demo showcases how to define a solid hyperelastic material
# using eigenvalues of the Cauchy strain tensor.

# In particular this demo emphasizes
# 1. symbolic computation of eigenvalues of a tensor field,
# 2. automatic differentiation of a strain energy function defined in terms of eigenvalues of the
# strain tensor.

# ---

# The task is to find a displacement field $u \in [H^1_0(\Omega)]^3$ which solves the variational
# problem (principle of virtual displacements)

# $$
# - \frac 12 \int_\Omega \delta C \colon \left(S_\text{bulk} + S_\text{shear}\right) \, \mathrm dx
# + \int_\Gamma \delta u \cdot t \, \mathrm ds = 0
# $$

# for all $\delta u \in [H^1_0(\Omega)]^3$. Stress tensor (second Piola-Kirchhoff) is computed from
# bulk and shear strain energies which are defined for a compressible Neo-Hookean material
# [https://doi.org/10.1177/1081286514544258] as

# $$
# \begin{align}
#     W_\text{bulk} &= \int_\Omega \frac{\kappa}{2} (J - 1)^2 \, \mathrm dx, \\
#     W_\text{shear} &= \int_\Omega \frac{\mu}{2} (I_1 - 3 - 2 \log J) \, \mathrm dx,
# \end{align}
# $$
# with Cauchy strain tensor $C = F^T F$, deformation gradient $F = I + \nabla u$ and its derived
# invariants $I_1 = \text{Tr}(C)$ and $J = \sqrt{\det C} = \det F$ and material parameters
# $\kappa$ and $\mu$.

# For the hyperelastic material the strain energy $W = W_\text{bulk} + W_\text{shear}$ is a
# potential for the stress tensor, i.e.

# $$ S = S_\text{bulk} + S_\text{shear} = 2 \frac{\partial W}{\partial C}. $$

# For isotropic strain energy functionals (which the Neo-Hookean model fulfils) we can write

# $$ s_i = 2 \frac{\partial W}{\partial c_i}, \quad i \in \{1, 2, 3\},$$

# for the three principal stresses (eigenvalues of the stress tensor $S$) $s_i$ and three principal
# stretches (eigenvalues of the strain tensor $C$) $c_i$. Using the notion of principal stresses and
# stretches, the inner product contraction in the above variational problem simplifies to

# $$
# - \frac 12 \int_\Omega \sum_i \delta c_i s_i \, \mathrm dx
# + \int_\Gamma \delta u \cdot t \, \mathrm ds = 0.
# $$

# The key point in the above is to express the strain energy functional purely in terms of
# principal stretches $c_i$, which is achieved using
# $$
# I_1 = c_0 + c_1 + c_2, \quad J = \sqrt{c_0 c_1 c_2}.
# $$
# Principal stretches $c_i$ are available as symbolic closed-form expression of the primary unknown
# displacement $u$ thanks to helper function `dolfiny.invariants.eigenstate`,
# see [https://doi.org/10.48550/arXiv.2111.02117] for more detail.

# %% tags=["hide-input"]
import argparse

from mpi4py import MPI
from petsc4py import PETSc

import basix
import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import numpy as np
import pyvista
import sympy.physics.units as syu

import dolfiny
from dolfiny.units import Quantity

parser = argparse.ArgumentParser(
    description="Solid elasticity with classic or spectral formulation"
)
parser.add_argument(
    "--formulation",
    choices=["classic", "spectral"],
    default="spectral",
    help="Choose strain formulation: classic (Cauchy strain) or spectral (principal stretches)",
    required=False,
)
args, unknown = parser.parse_known_args()


def mesh_tube3d_gmshapi(
    name="tube3d",
    r=1.0,
    t=0.2,
    h=1.0,
    nr=30,
    nt=6,
    nh=10,
    x0=0.0,
    y0=0.0,
    z0=0.0,
    do_quads=False,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 3d tube using the Python API of Gmsh.
    """
    tdim = 3  # target topological dimension

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.set_number("General.NumThreads", 1)  # reproducibility

        if do_quads:
            gmsh.option.set_number("Mesh.Algorithm", 8)
            gmsh.option.set_number("Mesh.Algorithm3D", 10)
            # gmsh.option.set_number("Mesh.SubdivisionAlgorithm", 2)
        else:
            gmsh.option.set_number("Mesh.Algorithm", 5)
            gmsh.option.set_number("Mesh.Algorithm3D", 4)
            gmsh.option.set_number("Mesh.AlgorithmSwitchOnFailure", 6)

        # Add model under given name
        gmsh.model.add(name)

        # Create points and line
        p0 = gmsh.model.occ.add_point(x0 + r, y0, 0.0)
        p1 = gmsh.model.occ.add_point(x0 + r + t, y0, 0.0)
        l0 = gmsh.model.occ.add_line(p0, p1)
        s0 = gmsh.model.occ.revolve(
            [(1, l0)],
            x0,
            y0,
            z0,
            0,
            0,
            1,
            angle=+gmsh.pi,
            numElements=[nr],
            recombine=do_quads,
        )
        s1 = gmsh.model.occ.revolve(
            [(1, l0)],
            x0,
            y0,
            z0,
            0,
            0,
            1,
            angle=-gmsh.pi,
            numElements=[nr],
            recombine=do_quads,
        )
        ring, _ = gmsh.model.occ.fuse([s0[1]], [s1[1]])
        tube = gmsh.model.occ.extrude(ring, 0, 0, h, [nh], recombine=do_quads)  # noqa: F841

        # Synchronize
        gmsh.model.occ.synchronize()

        # Get model entities
        _points, _lines, surfaces, volumes = (gmsh.model.occ.get_entities(d) for d in [0, 1, 2, 3])
        boundaries = gmsh.model.get_boundary(volumes, oriented=False)  # noqa: F841

        # Assertions, problem-specific
        assert len(volumes) == 2

        # Helper
        def extract_tags(a):
            return list(ai[1] for ai in a)

        # Extract certain tags, problem-specific
        tag_subdomains_total = extract_tags(volumes)

        # Set geometrical identifiers (obtained by inspection)
        tag_interfaces_lower = extract_tags([surfaces[0], surfaces[1]])
        tag_interfaces_upper = extract_tags([surfaces[6], surfaces[9]])
        tag_interfaces_inner = extract_tags([surfaces[2], surfaces[7]])
        tag_interfaces_outer = extract_tags([surfaces[3], surfaces[8]])

        # Define physical groups for subdomains (! target tag > 0)
        domain = 1
        gmsh.model.add_physical_group(tdim, tag_subdomains_total, domain)
        gmsh.model.set_physical_name(tdim, domain, "domain")

        # Define physical groups for interfaces (! target tag > 0)
        surface_lower = 1
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_lower, surface_lower)
        gmsh.model.set_physical_name(tdim - 1, surface_lower, "surface_lower")
        surface_upper = 2
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_upper, surface_upper)
        gmsh.model.set_physical_name(tdim - 1, surface_upper, "surface_upper")
        surface_inner = 3
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_inner, surface_inner)
        gmsh.model.set_physical_name(tdim - 1, surface_inner, "surface_inner")
        surface_outer = 4
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_outer, surface_outer)
        gmsh.model.set_physical_name(tdim - 1, surface_outer, "surface_outer")

        # Set refinement in radial direction
        gmsh.model.mesh.setTransfiniteCurve(l0, numNodes=nt)

        # Generate the mesh
        gmsh.model.mesh.generate()

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


class Xdmf3Reader(pyvista.XdmfReader):
    _vtk_module_name = "vtkIOXdmf3"
    _vtk_class_name = "vtkXdmf3Reader"


def plot_tube3d_pyvista(u, s, comm=MPI.COMM_WORLD):
    if comm.rank > 0:
        return

    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))

    plotter = pyvista.Plotter(off_screen=True, theme=dolfiny.pyvista.theme)
    plotter.add_axes()
    plotter.enable_parallel_projection()

    grid.point_data["u"] = u.x.array.reshape(-1, 3)
    grid.point_data["von_mises"] = s.x.array / 1e6  # to MPa

    grid_warped = grid.warp_by_vector("u", factor=1.0)

    if not grid.get_cell(0).is_linear:
        levels = 4
    else:
        levels = 0

    s = plotter.add_mesh(
        grid_warped.extract_surface(nonlinear_subdivision=levels),
        scalars="von_mises",
        scalar_bar_args={"title": "von Mises stress [MPa]"},
        n_colors=10,
    )

    s.mapper.scalar_range = [0.0, 0.6]

    plotter.add_mesh(
        grid_warped.separate_cells()
        .extract_surface(nonlinear_subdivision=levels)
        .extract_feature_edges(),
        style="wireframe",
        color="black",
        line_width=dolfiny.pyvista.pixels // 1000,
        render_lines_as_tubes=True,
    )

    plotter.show_axes()
    plotter.show()


# %% [markdown]
# This demo is parametrised by the choice of formulation: "classic" or "spectral".
# The "classic" formulation uses strain energy defined in terms of main invariants of
# the Cauchy strain tensor, while the "spectral" formulation uses strain energy defined
# in terms of the eigenvalues.

# %% tags=["hide-input"]
print(f"Arguments: {args}")

# %% [markdown]
# ## Meshing, boundary tagging and function spaces definition
#
# Mesh in this example is a tube with radius $r = 0.4$,
# thickness $t = 0.1$ and height $h = 1$.
# It is tesselated using 27 node quadratic hexahedral elements.
#
# Bottom of the tube is marked as "surface_lower" and top is marked as
# "surface_upper".
# %% tags=["hide-output"]
name = f"solid_elasticity_{args.formulation}"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
r, t, h = 0.4, 0.1, 1
nr, nt, nh = 16, 5, 8

# Create the regular mesh of a tube with given dimensions
gmsh_model, tdim = mesh_tube3d_gmshapi(name, r, t, h, nr, nt, nh, do_quads=True, order=2)

# Get mesh and meshtags
mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh_model, comm, rank=0)
mesh = mesh_data.mesh

# Define shorthands for labelled tags
surface_lower = mesh_data.physical_groups["surface_lower"].tag
surface_upper = mesh_data.physical_groups["surface_upper"].tag

# %% [markdown]
# Quadrature rule is limited to the 4th degree for performance reasons. The
# symbolic expressions resulting from the eigenvalues of the Cauchy strain tensor are rather
# involved so the time to assemble the forms increases.
#
# Function space discretisation is based on vector-valued isoparametric continuous Lagrange element
# with three components - since we model the displacement in three dimensions.

# %%
dx = ufl.Measure(
    "dx", domain=mesh, subdomain_data=mesh_data.cell_tags, metadata={"quadrature_degree": 4}
)
ds = ufl.Measure(
    "ds", domain=mesh, subdomain_data=mesh_data.facet_tags, metadata={"quadrature_degree": 4}
)

Ue = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
Uf = dolfinx.fem.functionspace(mesh, Ue)

print(f"Degrees-of-freedom per element: {Uf.element.space_dimension}")

# Define functions
u = dolfinx.fem.Function(Uf, name="u")
u_ = dolfinx.fem.Function(Uf, name="u_")  # boundary conditions

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(Uf))
(δu,) = δm

# Define state as (ordered) list of functions
m = [u]

# Functions for output / visualisation
vorder = mesh.geometry.cmap.degree
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder, (3,))), name="u")
so = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("P", vorder)), name="s")  # for output
# %%
plot_tube3d_pyvista(uo, so)

# %% [markdown]
# ## Material parameters with units
#
# There are four dimensional quantities in the problem:
#
# | Parameter | Value | Description |
# |-----------|-------|-------------|
# | $l_\text{ref}$ | $0.1\,\mathrm{m}$ | reference length scale |
# | $t_\text{ref}$ | $0.2\,\mathrm{MPa}$ | load scale |
# | $\mu$ | $\frac{E}{2(1 + \nu)}$ | shear modulus |
# | $\kappa$ | $\lambda + \frac{2}{3} \mu$ | bulk modulus |
#
# derived from Poisson ratio $\nu = 0.4$, Lamé coefficient
# $\lambda = \frac{E \nu}{(1 + \nu)(1 - 2 \nu)}$ and Young's modulus
# $E = 1 \, \mathrm{MPa}$.
#
# We can execute the Buckingham Pi analysis which shows overview of the
# dimensional quantities and derives a set of dimensionless numbers.
# In this examaple we arrive at two dimensionless numbers:
#
# 1. bulk-to-shear ratio $\kappa / \mu = 4.67$ and
# 2. loading factor $t_\text{ref} / \mu = 0.56$.
#
# %%
nu = 0.4
E = Quantity(mesh, 1, syu.mega * syu.pascal, "E")  # Young's modulus
mu = Quantity(mesh, E.scale / (2 * (1 + nu)), syu.mega * syu.pascal, "μ")  # shear modulus
λ = Quantity(
    mesh, E.scale * nu / ((1 + nu) * (1 - 2 * nu)), syu.mega * syu.pascal, "λ"
)  # Lamé constant
kappa = Quantity(mesh, λ.scale + 2 / 3 * mu.scale, syu.mega * syu.pascal, "κ")  # Lamé constant

l_ref = Quantity(mesh, 0.1, syu.meter, "l_ref")
t_ref = Quantity(mesh, 0.2, syu.mega * syu.pascal, "t_ref")

quantities = [l_ref, t_ref, mu, kappa]
quantities = [mu, kappa, l_ref, t_ref]
if comm.rank == 0:
    dolfiny.units.buckingham_pi_analysis(quantities)

# %% [markdown]
# ## Weak form

# %%
F = ufl.Identity(3) + ufl.grad(u)

# Strain measure: Cauchy strain tensor
C = F.T * F
C = ufl.variable(C)


def strain_energy_bulk(i1, i2, i3):
    J = ufl.sqrt(i3)
    return kappa / 2 * (J - 1) ** 2


def strain_energy_shear(i1, i2, i3):
    J = ufl.sqrt(i3)
    return mu / 2 * (i1 - 3 - 2 * ufl.ln(J))


def von_mises_stress(S):
    return ufl.sqrt(3 / 2 * ufl.inner(ufl.dev(S), ufl.dev(S)))


# Formulation-specific strain measures
if args.formulation == "spectral":
    c, _ = dolfiny.invariants.eigenstate(C)
    c = ufl.as_vector(c)
    c = ufl.variable(c)

    # Reconstruct the principal invariants from the principal stretches
    i1, i2, i3 = c[0] + c[1] + c[2], c[0] * c[1] + c[1] * c[2] + c[0] * c[2], c[0] * c[1] * c[2]

    δC = ufl.derivative(c, m, δm)
    S_bulk = 2 * ufl.diff(strain_energy_bulk(i1, i2, i3), c)
    S_shear = 2 * ufl.diff(strain_energy_shear(i1, i2, i3), c)
    S = S_bulk + S_shear

    svm = von_mises_stress(ufl.diag(S))
elif args.formulation == "classic":
    δC = ufl.derivative(C, m, δm)

    i1, i2, i3 = dolfiny.invariants.invariants_principal(C)

    S_bulk = 2 * ufl.diff(strain_energy_bulk(i1, i2, i3), C)
    S_shear = 2 * ufl.diff(strain_energy_shear(i1, i2, i3), C)
    S = S_bulk + S_shear

    svm = von_mises_stress(S)
else:
    raise RuntimeError(f"Unknown formulation '{args.formulation}'")

# %% [markdown]
# Boundary traction $t$ is created to represent rotational vector field in the shifted $xy$-plane
# which is scaled with the reference load scale $t_\text{ref}$.
# We first compute radial vector field in the shifted $xy$-plane
# $$d = x - l_\text{ref} h (0,0,1)^T$$
# which we normalize and cross product with the unit vector $e_z = (0,0,1)^T$,
# $$ t = \alpha t_\text{ref} \frac{d}{||d||} \times e_z.$$
# A load factor $\alpha$ is increased from 0 to 1 during the loading procedure.

# %%
x0 = ufl.SpatialCoordinate(mesh)
load_factor = dolfinx.fem.Constant(mesh, scalar(0.0))
ez = ufl.as_vector([0.0, 0.0, 1.0])
d = x0 - l_ref * h * ez
d /= ufl.sqrt(ufl.inner(d, d))
t = ufl.cross(d, ez) * t_ref * load_factor

mapping = {
    mesh.ufl_domain(): l_ref,
    u: l_ref * u,
    δu: l_ref * δu,
}

terms = {
    "int_bulk": -1 / 2 * ufl.inner(δC, S_bulk) * dx,
    "int_shear": -1 / 2 * ufl.inner(δC, S_shear) * dx,
    "external": ufl.inner(δu, t) * ds(surface_upper),
}
factorized = dolfiny.units.factorize(terms, quantities, mode="factorize", mapping=mapping)
assert isinstance(factorized, dict)

dimsys = syu.si.SI.get_dimension_system()
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(terms["int_bulk"], quantities, mapping),
    syu.energy,
)
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(strain_energy_bulk(i1, i2, i3), quantities, mapping),
    syu.energy * syu.length**-3,
)
assert dimsys.equivalent_dims(
    dolfiny.units.get_dimension(S_shear, quantities, mapping),
    syu.pressure,
)

reference_term = "int_bulk"
ref_factor = factorized[reference_term].factor

normalized = dolfiny.units.normalize(factorized, reference_term, quantities)
form = sum(normalized.values(), ufl.form.Zero())

# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# %% [markdown]
# The problem solved leads to symmetric positive definite system on the algebraic level.
# We choose to solve it using MUMPS Cholesky $LDL^T$ solver for general symmetric matrices.
# We explicitly numerical pivoting by setting CNTL(1) = 0.

# The nonlinear SNES solver is configured to use Newton line search with
# no (basic) line search.

# %%
opts = PETSc.Options(name)  # type: ignore[attr-defined]
opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 1.0e-08
opts["snes_max_it"] = 10
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_cntl_1"] = 0.0

# %% [markdown]
# ```{note}
#    Compilation of complicated eigenvalue expressions could take considerable time,
#    especially for the ARM64 architecture. We disable selected optimizations.
#    Alternatively, we can disable all optimizations by setting `-O0`. We found this
#    approach useful for fast model development and testing when compilation time matters.
#    However, for production runs we recommend to enable optimizations e.g. keep the
#    default `-O2` (or even `-O3`).
# ```

# %% tags=["hide-input", "hide-output"]
# FFCx options (formulation-specific)
if args.formulation == "spectral":
    # ARM64-specific optimizations for spectral formulation
    jit_options = dict(
        cffi_extra_compile_args=[
            "-fdisable-rtl-combine",
            "-fno-schedule-insns",
            "-fno-schedule-insns2",
        ]
    )
else:
    # Standard options for classic formulation
    jit_options = dict(cffi_extra_compile_args=["-g0"])

# Create nonlinear problem: SNES
problem = dolfiny.snesproblem.SNESProblem(forms, m, prefix=name, jit_options=jit_options)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
b_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, mesh_data.facet_tags, surface_lower)

# Set/update boundary conditions
problem.bcs = [
    dolfinx.fem.dirichletbc(u_, b_dofs_Uf),  # u lower face
]

# %% tags=["hide-output"]
# Apply external force via load stepping
for lf in np.linspace(0.0, 1.0, 10 + 1)[1:]:
    # Set load factor
    load_factor.value = lf
    dolfiny.utils.pprint(f"\n*** Load factor = {lf:.4f} ({args.formulation} formulation) \n")

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Assert symmetry of operator
    assert dolfiny.la.is_symmetric(problem.J)

# Interpolate for output purposes
dolfiny.interpolation.interpolate(u, uo)
dolfiny.interpolation.interpolate(svm, so)

# Write results to file
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh)
    ofile.write_function(uo)
    ofile.write_function(so)

# %%
plot_tube3d_pyvista(uo, so)

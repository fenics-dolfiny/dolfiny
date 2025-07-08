#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl
from dolfinx import default_scalar_type as scalar

import numpy as np

import dolfiny

"""
First order nonlinear system of ODEs: (Duffing oscillator, undamped, unforced)

(1) dot v + s = 0
(2) dot s - st(u, v) = 0

(*) dot u - v = 0

with the constitutive law: s(u) = [a + b * u^2] * u
and its rate form: st(u, v) = [a + 3 * b * u^2] * v
and initial conditions: u(t=0) = u_0, v(t=0) = v_0 and s(t=0) = s(u_0)
"""

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)

# Problem parameters, note: (a + b * u_0**2) !> 0
a, b = 0.3, -0.25

# Initial conditions
u_0, v_0 = 0.8, 0.0


def _s(u):
    return (a + b * u**2) * u  # constitutive law


def _st(u, v):
    return (a + 3 * b * u**2) * v  # rate of constitutive law


V = dolfinx.fem.functionspace(mesh, ("DP", 0))
S = dolfinx.fem.functionspace(mesh, ("DP", 0))

v = dolfinx.fem.Function(V, name="v")
s = dolfinx.fem.Function(S, name="s")
vt = dolfinx.fem.Function(V, name="vt")
st = dolfinx.fem.Function(S, name="st")

u = dolfinx.fem.Function(V, name="u")
d = dolfinx.fem.Function(V, name="d")  # dummy

δm = ufl.TestFunctions(ufl.MixedFunctionSpace(V, S))
δv, δs = δm

m, mt = [v, s], [vt, st]

# Set initial conditions
v.x.petsc_vec.set(v_0), vt.x.petsc_vec.set(-_s(u_0))
s.x.petsc_vec.set(_s(u_0)), st.x.petsc_vec.set(_st(u_0, v_0))
u.x.petsc_vec.set(u_0)

[w.x.petsc_vec.ghostUpdate() for w in [v, s, u, vt, st]]

# Measure
dx = ufl.Measure("dx", domain=mesh)

# Number of time steps
nT = 100

# Global time
t = dolfinx.fem.Constant(mesh, scalar(0.0))

# Time step size
dt = dolfinx.fem.Constant(mesh, scalar(25 / nT))

# Time integrator
odeint = dolfiny.odeint.ODEInt(t=t, dt=dt, x=m, xt=mt)

# Expression for time-integrated quantities
u_expr = u + odeint.integral_dt(v)

# Strong form residuals
r1 = vt + s
r2 = st - _st(u_expr, v)

# Weighted residual (as one-form)
form = δv * r1 * dx + δs * r2 * dx

# Overall form (as one-form)
form = odeint.discretise_in_time(form)
# Overall form (as list of forms)
forms = ufl.extract_blocks(form)

# Options for PETSc backend
opts = PETSc.Options()  # type: ignore[attr-defined]

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-12

# Create nonlinear problem
problem = dolfiny.snesproblem.SNESProblem(forms, m)

# Book-keeping of results
v_, vt_, s_, st_, u_ = (np.zeros(nT + 1) for w in [v, vt, s, st, u])
v_[0], vt_[0], s_[0], st_[0], u_[0] = (
    w.x.petsc_vec.sum() / w.x.petsc_vec.getSize() for w in [v, vt, s, st, u]
)

# Process time steps
for ts in range(1, nT + 1):
    dolfiny.utils.pprint(
        f"\n+++ Processing time instant = {t.value + dt.value:7.3f} in step {ts:d}\n"
    )

    # Stage next time step
    odeint.stage()

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    problem.status(verbose=True, error_on_failure=True)

    # Update solution states for time integration
    odeint.update()

    # Assert zero residual at t + dt
    assert np.isclose(dolfiny.expression.assemble(r1, dx), 0.0, atol=1e-10), (
        "Non-zero residual r1 at (t + dt)!"
    )
    assert np.isclose(dolfiny.expression.assemble(r2, dx), 0.0, atol=1e-10), (
        "Non-zero residual r2 at (t + dt)!"
    )

    # Assign time-integrated quantities
    dolfiny.interpolation.interpolate(u_expr, d)
    dolfiny.interpolation.interpolate(d, u)

    # Store results
    v_[ts], vt_[ts], s_[ts], st_[ts], u_[ts] = (
        w.x.petsc_vec.sum() / w.x.petsc_vec.getSize() for w in [v, vt, s, st, u]
    )


# Compare with reference solution
import sys  # noqa: E402

sys.path.append("../../test")
from standalone_odeint import (  # noqa: E402
    ode_2nd_nonlinear_closed,
    ode_2nd_nonlinear_odeint,
)

u_c, v_c, a_c = u_, v_, vt_
u_e, v_e, a_e = ode_2nd_nonlinear_closed(a, b, u_0, nT=nT, dt=dt.value)
u_z, v_z, a_z = ode_2nd_nonlinear_odeint(a, b, u_0, nT=nT, dt=dt.value)

if MPI.COMM_WORLD.rank == 0:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    ax1.set_title("Duffing oscillator: velocity-force vs. displacement formulation", fontsize=12)
    ax1.set_xlabel(r"time $t$", fontsize=12)
    ax1.set_ylabel(r"solution", fontsize=12)
    ax1.grid(linewidth=0.25)
    fig.tight_layout()

    tt = np.linspace(0, dt.value * nT, nT + 1)

    # u
    ax1.plot(tt, u_e, color="tab:olive", linestyle="-", linewidth=1.0, label="u(t) exact")
    ax1.plot(
        tt,
        u_c,
        color="tab:red",
        linestyle="",
        markersize=3.0,
        marker="o",
        label="u(t) 2x 1st order ODE",
    )
    ax1.plot(
        tt,
        u_z,
        color="tab:blue",
        linestyle="",
        markersize=1.5,
        marker="s",
        label="u(t) 1x 2nd order ODE",
    )
    # v
    ax1.plot(tt, v_e, color="tab:cyan", linestyle="-", linewidth=1.0, label="v(t) exact")
    ax1.plot(
        tt,
        v_c,
        color="tab:red",
        linestyle="",
        markersize=3.0,
        marker="o",
        label="v(t) 2x 1st order ODE",
    )
    ax1.plot(
        tt,
        v_z,
        color="tab:blue",
        linestyle="",
        markersize=1.5,
        marker="s",
        label="v(t) 1x 2nd order ODE",
    )
    # a
    ax1.plot(tt, a_e, color="tab:pink", linestyle="-", linewidth=1.0, label="a(t) exact")
    ax1.plot(
        tt,
        a_c,
        color="tab:red",
        linestyle="",
        markersize=3.0,
        marker="o",
        label="a(t) 2x 1st order ODE",
    )
    ax1.plot(
        tt,
        a_z,
        color="tab:blue",
        linestyle="",
        markersize=1.5,
        marker="s",
        label="a(t) 1x 2nd order ODE",
    )
    # s
    ax1.plot(
        tt, _s(u_e), color="tab:orange", linestyle="-", linewidth=1.0, marker="", label="s(t) exact"
    )
    ax1.plot(
        tt,
        s_,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        marker="",
        label="s(t) 2x 1st order ODE",
    )
    ax1.plot(
        tt,
        _s(u_z),
        color="tab:green",
        linestyle=":",
        linewidth=1.0,
        marker="",
        label="s(t) 1x 2nd order ODE",
    )

    ax1.legend(loc="lower center", ncol=4, fontsize="xx-small")
    fig.savefig("duffing_solution.pdf")

from mpi4py import MPI

import dolfinx
import ufl

import numpy as np

import dolfiny

# === ODEInt-based solutions =================================================


def ode_1st_linear_odeint(a=1.0, b=0.5, u_0=1.0, nT=100, dt=0.01, **kwargs):
    """
    Create 1st order ODE problem and solve with `ODEInt` time integrator.

    First order linear ODE:
    dot u + a * u - b = 0 with initial condition u(t=0) = u_0
    """

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    U = dolfinx.fem.functionspace(mesh, ("DP", 0))

    u = dolfinx.fem.Function(U, name="u")
    ut = dolfinx.fem.Function(U, name="ut")

    u.x.petsc_vec.set(u_0)  # initial condition
    ut.x.petsc_vec.set(b - a * u_0)  # exact initial rate of this ODE for generalised alpha

    u.x.petsc_vec.ghostUpdate()
    ut.x.petsc_vec.ghostUpdate()

    δu = ufl.TestFunction(U)

    dx = ufl.Measure("dx", domain=mesh)

    # Global time
    time = dolfinx.fem.Constant(mesh, 0.0)

    # Time step size
    dt = dolfinx.fem.Constant(mesh, dt)

    # Time integrator
    odeint = dolfiny.odeint.ODEInt(t=time, dt=dt, x=u, xt=ut, **kwargs)

    # Resdiual form (as one-form)
    r = ut + a * u - b

    # Weighted residual form (as one-form)
    f = δu * r * dx

    # Overall form (as one-form)
    F = odeint.discretise_in_time(f)
    # Overall form (as list of forms)
    F = dolfiny.function.extract_blocks(F, [δu])

    # Create problem (although having a linear ODE we use the dolfiny.snesblockproblem API)
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, [u])

    # Silence SNES monitoring during test
    problem.verbose = dict(snes=False, ksp=False)

    # Book-keeping of results
    u_, ut_ = np.zeros(nT + 1), np.zeros(nT + 1)
    u_[0], ut_[0] = (v.x.petsc_vec.sum() / v.x.petsc_vec.getSize() for v in [u, ut])

    dolfiny.utils.pprint(f"+++ Processing time steps = {nT}")

    # Process time steps
    for time_step in range(1, nT + 1):
        # Stage next time step
        odeint.stage()

        # Solve (linear) problem
        problem.solve()

        # Update solution states for time integration
        odeint.update()

        # Assert zero residual at t + dt
        assert np.isclose(dolfiny.expression.assemble(r, dx), 0.0, atol=1e-6), (
            "Non-zero residual at (t + dt)!"
        )

        # Store results
        u_[time_step], ut_[time_step] = (
            v.x.petsc_vec.sum() / v.x.petsc_vec.getSize() for v in [u, ut]
        )

    return u_, ut_


def ode_1st_nonlinear_odeint(a=2.0, b=1.0, c=8.0, nT=100, dt=0.01, **kwargs):
    """
    Create 1st order ODE problem and solve with `ODEInt` time integrator.

    First order nonlinear non-autonomous ODE:
    t * dot u - a * cos(c*t) * u^2 - 2 * u - a * b^2 * t^4 * cos(c*t) = 0
                                        with initial condition u(t=1) = 0
    """

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    U = dolfinx.fem.functionspace(mesh, ("DP", 0))

    u = dolfinx.fem.Function(U, name="u")
    ut = dolfinx.fem.Function(U, name="ut")

    u.x.petsc_vec.set(0.0)  # initial condition
    ut.x.petsc_vec.set(a * b**2 * np.cos(c))  # exact initial rate of this ODE for generalised alpha

    u.x.petsc_vec.ghostUpdate()
    ut.x.petsc_vec.ghostUpdate()

    δu = ufl.TestFunction(U)

    dx = ufl.Measure("dx", domain=mesh)

    # Global time
    t = dolfinx.fem.Constant(mesh, 1.0)

    # Time step size
    dt = dolfinx.fem.Constant(mesh, dt)

    # Time integrator
    odeint = dolfiny.odeint.ODEInt(t=t, dt=dt, x=u, xt=ut, **kwargs)

    # Strong form residual (as one-form)
    r = t * ut - a * ufl.cos(c * t) * u**2 - 2 * u - a * b**2 * t**4 * ufl.cos(c * t)

    # Weighted residual (as one-form)
    f = δu * r * dx

    # Overall form (as one-form)
    F = odeint.discretise_in_time(f)
    # Overall form (as list of forms)
    F = dolfiny.function.extract_blocks(F, [δu])

    # Options for PETSc backend
    from petsc4py import PETSc

    opts = PETSc.Options()
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_atol"] = 1.0e-09
    opts["snes_rtol"] = 1.0e-12

    # Create nonlinear problem
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, [u])

    # Silence SNES monitoring during test
    problem.verbose = dict(snes=False, ksp=False)

    # Book-keeping of results
    u_, ut_ = np.zeros(nT + 1), np.zeros(nT + 1)
    u_[0], ut_[0] = (v.x.petsc_vec.sum() / v.x.petsc_vec.getSize() for v in [u, ut])

    dolfiny.utils.pprint(f"+++ Processing time steps = {nT}")

    # Process time steps
    for time_step in range(1, nT + 1):
        # Stage next time step
        odeint.stage()

        # Solve nonlinear problem
        problem.solve()

        # Assert convergence of nonlinear solver
        assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

        # Update solution states for time integration
        odeint.update()

        # Assert zero residual at t + dt
        assert np.isclose(dolfiny.expression.assemble(r, dx), 0.0, atol=1e-6), (
            "Non-zero residual at (t + dt)!"
        )

        # Store results
        u_[time_step], ut_[time_step] = (
            v.x.petsc_vec.sum() / v.x.petsc_vec.getSize() for v in [u, ut]
        )

    return u_, ut_


def ode_2nd_linear_odeint(a=12.0, b=1000.0, c=1000.0, u_0=0.5, du_0=0.0, nT=100, dt=0.01, **kwargs):
    """
    Create 2nd order ODE problem and solve with `ODEInt` time integrator.

    Second order linear ODE:
    ddot u + a * dot u + b * u - c = 0 with initial conditions u(t=0) = u_0 ; du(t=0) = du_0
    """

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    U = dolfinx.fem.functionspace(mesh, ("DP", 0))

    u = dolfinx.fem.Function(U, name="u")
    ut = dolfinx.fem.Function(U, name="ut")
    utt = dolfinx.fem.Function(U, name="utt")

    u.x.petsc_vec.set(u_0)  # initial condition
    ut.x.petsc_vec.set(du_0)  # initial condition
    utt.x.petsc_vec.set(
        c - a * du_0 - b * u_0
    )  # exact initial rate of rate of this ODE for generalised alpha

    u.x.petsc_vec.ghostUpdate()
    ut.x.petsc_vec.ghostUpdate()
    utt.x.petsc_vec.ghostUpdate()

    δu = ufl.TestFunction(U)

    dx = ufl.Measure("dx", domain=mesh)

    # Global time
    time = dolfinx.fem.Constant(mesh, 0.0)

    # Time step size
    dt = dolfinx.fem.Constant(mesh, dt)

    # Time integrator
    odeint = dolfiny.odeint.ODEInt2(t=time, dt=dt, x=u, xt=ut, xtt=utt, **kwargs)

    # Strong form residual (as one-form)
    r = utt + a * ut + b * u - c

    # Weighted residual (as one-form)
    f = δu * r * dx

    # Overall form (as one-form)
    F = odeint.discretise_in_time(f)
    # Overall form (as list of forms)
    F = dolfiny.function.extract_blocks(F, [δu])

    # Create problem (although having a linear ODE we use the dolfiny.snesblockproblem API)
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, [u])

    # Silence SNES monitoring during test
    problem.verbose = dict(snes=False, ksp=False)

    # Book-keeping of results
    u_, ut_, utt_ = np.zeros(nT + 1), np.zeros(nT + 1), np.zeros(nT + 1)
    u_[0], ut_[0], utt_[0] = (v.x.petsc_vec.sum() / v.x.petsc_vec.getSize() for v in [u, ut, utt])

    dolfiny.utils.pprint(f"+++ Processing time steps = {nT}")

    # Process time steps
    for time_step in range(1, nT + 1):
        # Stage next time step
        odeint.stage()

        # Solve (linear) problem
        problem.solve()

        # Update solution states for time integration
        odeint.update()

        # Assert zero residual at t + dt
        assert np.isclose(dolfiny.expression.assemble(r, dx), 0.0, atol=1e-6), (
            "Non-zero residual at (t + dt)!"
        )

        # Store results
        u_[time_step], ut_[time_step], utt_[time_step] = (
            v.x.petsc_vec.sum() / v.x.petsc_vec.getSize() for v in [u, ut, utt]
        )

    return u_, ut_, utt_


def ode_2nd_nonlinear_odeint(a=100, b=-50, u_0=1.0, nT=100, dt=0.01, **kwargs):
    """
    Create 2nd order ODE problem and solve with `ODEInt` time integrator.

    Second order nonlinear ODE: (Duffing)
    ddot u + a * u + b * u^3 = 0 with initial conditions u(t=0) = u_0 ; du(t=0) = 0
    """

    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    U = dolfinx.fem.functionspace(mesh, ("DP", 0))

    u = dolfinx.fem.Function(U, name="u")
    ut = dolfinx.fem.Function(U, name="ut")
    utt = dolfinx.fem.Function(U, name="utt")

    u.x.petsc_vec.set(u_0)  # initial condition
    ut.x.petsc_vec.set(0.0)  # initial condition
    utt.x.petsc_vec.set(
        -a * u_0 - b * u_0**3
    )  # exact initial rate of rate of this ODE for generalised alpha

    u.x.petsc_vec.ghostUpdate()
    ut.x.petsc_vec.ghostUpdate()
    utt.x.petsc_vec.ghostUpdate()

    δu = ufl.TestFunction(U)

    dx = ufl.Measure("dx", domain=mesh)

    # Global time
    time = dolfinx.fem.Constant(mesh, 0.0)

    # Time step size
    dt = dolfinx.fem.Constant(mesh, dt)

    # Time integrator
    odeint = dolfiny.odeint.ODEInt2(t=time, dt=dt, x=u, xt=ut, xtt=utt, **kwargs)

    # Strong form residual (as one-form)
    r = utt + a * u + b * u**3

    # Weighted residual (as one-form)
    f = δu * r * dx

    # Overall form (as one-form)
    F = odeint.discretise_in_time(f)
    # Overall form (as list of forms)
    F = dolfiny.function.extract_blocks(F, [δu])

    # Options for PETSc backend
    from petsc4py import PETSc

    opts = PETSc.Options()
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_atol"] = 1.0e-09
    opts["snes_rtol"] = 1.0e-12

    # Create nonlinear problem
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, [u])

    # Silence SNES monitoring during test
    problem.verbose = dict(snes=False, ksp=False)

    # Book-keeping of results
    u_, ut_, utt_ = np.zeros(nT + 1), np.zeros(nT + 1), np.zeros(nT + 1)
    u_[0], ut_[0], utt_[0] = (v.x.petsc_vec.sum() / v.x.petsc_vec.getSize() for v in [u, ut, utt])

    dolfiny.utils.pprint(f"+++ Processing time steps = {nT}")

    # Process time steps
    for time_step in range(1, nT + 1):
        # Stage next time step
        odeint.stage()

        # Solve nonlinear problem
        problem.solve()

        # Assert convergence of nonlinear solver
        assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

        # Update solution states for time integration
        odeint.update()

        # Assert zero residual at t + dt
        assert np.isclose(dolfiny.expression.assemble(r, dx), 0.0, atol=1e-6), (
            "Non-zero residual at (t + dt)!"
        )

        # Store results
        u_[time_step], ut_[time_step], utt_[time_step] = (
            v.x.petsc_vec.sum() / v.x.petsc_vec.getSize() for v in [u, ut, utt]
        )

    return u_, ut_, utt_


def ode_1st_nonlinear_mdof_odeint(a=100, b=-50, u_0=1.0, nT=100, dt=0.01, **kwargs):
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
    a, b = a, b

    # Initial conditions
    u_0, v_0 = u_0, 0.0

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

    δv = ufl.TestFunction(V)
    δs = ufl.TestFunction(S)

    m, mt, δm = [v, s], [vt, st], [δv, δs]

    # Set initial conditions
    v.x.petsc_vec.set(v_0), vt.x.petsc_vec.set(-_s(u_0))
    s.x.petsc_vec.set(_s(u_0)), st.x.petsc_vec.set(_st(u_0, v_0))
    u.x.petsc_vec.set(u_0)

    [w.x.petsc_vec.ghostUpdate() for w in [v, s, u, vt, st]]

    # Measure
    dx = ufl.Measure("dx", domain=mesh)

    # Number of time steps
    nT = nT

    # Global time
    t = dolfinx.fem.Constant(mesh, 0.0)

    # Time step size
    dt = dolfinx.fem.Constant(mesh, dt)

    # Time integrator
    odeint = dolfiny.odeint.ODEInt(t=t, dt=dt, x=m, xt=mt, **kwargs)

    # Expression for time-integrated quantities
    u_expr = u + odeint.integral_dt(v)

    # Strong form residuals
    r1 = vt + s
    r2 = st - _st(u_expr, v)

    # Weighted residual (as one-form)
    f = δv * r1 * dx + δs * r2 * dx

    # Overall form (as one-form)
    F = odeint.discretise_in_time(f)
    # Overall form (as list of forms)
    F = dolfiny.function.extract_blocks(F, δm)

    # Options for PETSc backend
    from petsc4py import PETSc

    opts = PETSc.Options()
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_atol"] = 1.0e-09
    opts["snes_rtol"] = 1.0e-12

    # Create nonlinear problem
    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m)

    # Silence SNES monitoring during test
    problem.verbose = dict(snes=False, ksp=False)

    # Book-keeping of results
    u_, v_, vt_ = (np.zeros(nT + 1) for w in [u, v, vt])
    u_[0], v_[0], vt_[0] = (w.x.petsc_vec.sum() / w.x.petsc_vec.getSize() for w in [u, v, vt])

    dolfiny.utils.pprint(f"+++ Processing time steps = {nT}")

    # Process time steps
    for ts in range(1, nT + 1):
        # Stage next time step
        odeint.stage()

        # Solve nonlinear problem
        problem.solve()

        # Assert convergence of nonlinear solver
        assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

        # Update solution states for time integration
        odeint.update()

        # Assert zero residual at t + dt
        assert np.isclose(dolfiny.expression.assemble(r1, dx), 0.0, atol=1e-6), (
            "Non-zero residual r1 at (t + dt)!"
        )
        assert np.isclose(dolfiny.expression.assemble(r2, dx), 0.0, atol=1e-6), (
            "Non-zero residual r2 at (t + dt)!"
        )

        # Assign time-integrated quantities
        dolfiny.interpolation.interpolate(u_expr, d)
        dolfiny.interpolation.interpolate(d, u)

        # Store results
        u_[ts], v_[ts], vt_[ts] = (
            w.x.petsc_vec.sum() / w.x.petsc_vec.getSize() for w in [u, v, vt]
        )

    return u_, v_, vt_


# === Closed-form solutions ==================================================


def ode_1st_linear_closed(a=1.0, b=0.5, u_0=1.0, nT=100, dt=0.01):
    """
    Solve ODE in closed form (analytically, at discrete time instances).

    First order linear ODE:
    dot u + a * u - b = 0 with initial condition u(t=0) = u_0
    """

    t = np.linspace(0, nT * dt, num=nT + 1)

    u = (u_0 - b / a) * np.exp(-a * t) + b / a
    ut = -a * (u_0 - b / a) * np.exp(-a * t)

    return u, ut


def ode_1st_nonlinear_closed(a=2.0, b=1.0, c=8.0, nT=100, dt=0.01):
    """
    Solve ODE in closed form (analytically, at discrete time instances).

    First order nonlinear non-autonomous ODE:
    t * dot u - a * cos(c*t) * u^2 - 2*u - a * b^2 * t^4 * cos(c*t) = 0
                                      with initial condition u(t=1) = 0
    """

    t = np.linspace(1, 1 + nT * dt, num=nT + 1)

    z = c * t * np.sin(c * t) - c * np.sin(c) + np.cos(c * t) - np.cos(c)
    zt = c**2 * t * np.cos(c * t)

    u = b * t**2 * np.tan(a * b / c**2 * z)
    ut = (
        2 * b * t * np.tan(a * b / c**2 * z)
        + a * b**2 / c**2 * t**2 * (np.tan(a * b / c**2 * z) ** 2 + 1) * zt
    )

    return u, ut


def ode_2nd_linear_closed(a=12.0, b=1000.0, c=1000.0, u_0=0.5, du_0=0.0, nT=100, dt=0.01):
    """
    Solve ODE in closed form (analytically, at discrete time instances).

    Second order linear ODE:
    ddot u + a * dot u + b * u - c = 0 with initial conditions u(t=0) = u_0 ; du(t=0) = du_0
    """

    t = np.linspace(0, nT * dt, num=nT + 1)

    La = np.sqrt(4 * b - a**2)
    C2 = u_0 - c / b
    C1 = 2 / La * (du_0 + C2 * a / 2)

    u = np.exp(-0.5 * a * t) * (C1 * np.sin(La / 2 * t) + C2 * np.cos(La / 2 * t)) + c / b
    ut = (
        np.exp(-a * t / 2)
        * ((C1 * La * np.cos(La * t / 2)) / 2 - (C2 * La * np.sin(La * t / 2)) / 2)
        - (a * np.exp(-a * t / 2) * (C2 * np.cos(La * t / 2) + C1 * np.sin(La * t / 2))) / 2
    )
    utt = (
        (a**2 * np.exp(-a * t / 2) * (C2 * np.cos(La * t / 2) + C1 * np.sin(La * t / 2))) / 4
        - np.exp(-a * t / 2)
        * (C2 * La**2 * np.cos(La * t / 2) / 4 + (C1 * La**2 * np.sin(La * t / 2)) / 4)
        - a
        * np.exp(-a * t / 2)
        * ((C1 * La * np.cos(La * t / 2)) / 2 - (C2 * La * np.sin(La * t / 2)) / 2)
    )

    return u, ut, utt


def ode_2nd_nonlinear_closed(a=100, b=-50, u_0=1.0, nT=100, dt=0.01):
    """
    Solve ODE in closed form (analytically, at discrete time instances).

    Second order nonlinear ODE: (Duffing)
    ddot u + a * u + b * u^3 = 0 with initial conditions u(t=0) = u_0 ; du(t=0) = 0
    """

    t = np.linspace(0, nT * dt, num=nT + 1)

    # Analytical solution in terms of Jacobi elliptic functions (exact)
    # u(t) = u_0 * cn(c * t, m)
    # ut(t) = -c * u_0 * dn(c * t, m) * sn(c * t, m)
    # utt(t) = -c^2 * u_0 * cn(c * t, m) * dn(c * t, m) * (dn(c * t, m)
    #          - m * sd(c * t, m) * sn(c * t, m))
    import scipy.special

    c = np.sqrt(a + b * u_0**2)
    k = b * u_0**2 / 2 / (a + b * u_0**2)
    # Capture cases for which the modulus k is not within the [0,1] interval
    # of `scipy.special.ellipj`.
    # This is needed for a softening Duffing oscillator with b < 0.
    # Arguments for `scipy.special.ellipj`
    u = c * t
    m = k
    #
    if m >= 0 and m <= 1:
        sn, cn, dn, _ = scipy.special.ellipj(u, m)
        sn_, cn_, dn_ = sn, cn, dn
    if m > 1:
        u_ = u * m ** (1 / 2)
        m_ = m ** (-1)
        sn, cn, dn, _ = scipy.special.ellipj(u_, m_)
        sn_, cn_, dn_ = m ** (-1 / 2) * sn, dn, cn
    if m < 0:
        u_ = u * (1 / (1 - m)) ** (-1 / 2)
        m_ = -m / (1 - m)
        sn, cn, dn, _ = scipy.special.ellipj(u_, m_)
        sn_, cn_, dn_ = (1 / (1 - m)) ** (1 / 2) * sn / dn, cn / dn, 1 / dn

    u = u_0 * cn_
    ut = -c * u_0 * dn_ * sn_
    utt = -(c**2) * u_0 * cn_ * dn_ * (dn_ - m * sn_ / dn_ * sn_)

    return u, ut, utt

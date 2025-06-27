import dolfinx

from dolfiny.snesproblem import SNESProblem
from dolfiny.utils import pprint


class Crisfield:
    def __init__(self, problem: SNESProblem, λ: dolfinx.fem.Constant, monitor=None, inner=None):
        self.problem = problem
        self.monitor = monitor
        self.inner = inner if inner is not None else lambda v1, v2: v1.dot(v2)

        self.λ = λ

        # copying from F or x picks up nest/block format
        self.dFdλ = problem.active_F.copy()  # linear-in-λ terms of residual (assumption!)
        self.dx = problem.active_x.copy()  # converged x increment towards last continuation step
        self.Δx = problem.active_x.copy()  # accumulated x increment of current continuation step
        self.δx_dFdλ = problem.active_x.copy()

        self.dλ = dolfinx.fem.Constant(
            self.λ._ufl_domain, self.λ.value
        )  # converged λ increment towards last continuation step
        self.Δλ = dolfinx.fem.Constant(
            self.λ._ufl_domain, self.λ.value
        )  # accumulated λ increment of current continuation step

        self.ds = 0.1  # continuation step size, default
        self.psi = 1.0  # continuation shape, default

        self.problem.snes.setUpdate(Crisfield.update, kargs=dict(continuation=self))

    def initialise(self, ds=None, λ=None, psi=None):
        if ds is not None:
            self.ds = ds

        if psi is not None:
            self.psi = λ

        if λ is not None:
            self.λ.value = λ

        # dx = 0.0
        with self.dx.localForm() as dx_local:
            dx_local.set(0.0)

        # dλ = ds
        self.dλ.value = self.ds

    def solve_step(self, ds=None, zero_x_predictor=False):
        if ds is not None:
            self.ds = ds

        # set step predictors
        # Δx
        if zero_x_predictor:
            # Δx = 0.0
            with self.Δx.localForm() as Δx_local:
                Δx_local.set(0.0)
        else:
            # Δx = dx (saves about one Newton iterate over zeroed Δx)
            with self.dx.localForm() as dx_local, self.Δx.localForm() as Δx_local:
                dx_local.copy(Δx_local)
        # Δλ
        self.Δλ.value = self.dλ.value

        # set solution predictors according to step predictors
        # x += Δx
        with self.problem.active_x.localForm() as x_local, self.Δx.localForm() as Δx_local:
            x_local += Δx_local
        # λ += Δλ
        self.λ.value += self.Δλ.value

        # solve non-linear block problem with SNES and custom update
        #
        # NOTE: internal workflow of SNES solver
        #
        # (prep) F(x), monitor, check |F|
        # loop:
        # (a) call update function "update":
        #     (a1) J(x) dy = dFdλ(x) [if not 1st iteration],
        #     (a2) solve arc-length equation,
        #     (a3) update x and λ,
        #     (a4) compute F(x, λ) and dFdλ(x)
        # (b) J(x)
        # (c) solve J(x) dx = F(x)
        # (d) x -= dx
        # (e) F(x)
        # (f) monitor
        # (g) check convergence
        #
        self.problem.solve()

        # Assert convergence of nonlinear solver
        self.problem.status(verbose=True, error_on_failure=True)

        # call update once again
        Crisfield.update(self.problem.snes, self.problem.snes.getIterationNumber(), self)

        # store converged accumulated increments of this continuation step
        # dx <- Δx
        with self.dx.localForm() as dx_local, self.Δx.localForm() as Δx_local:
            Δx_local.copy(dx_local)
        # dλ <- Δλ
        self.dλ.value = self.Δλ.value

    @staticmethod
    def update(snes, snes_iteration, continuation):
        """Crisfield's continuation update based on solving the quadratic arc-length equation.

            (Δx_k).T * (Δx_k) + (Δλ_k * dFdλ).T * (Δλ_k * dFdλ) = ds**2

        for δλ_k when incremental updates

            Δx_k = Δx_{k-1} + δx_k, with δx_k = δx_dFdλ * δλ_k, and
            Δλ_k = Δλ_{k-1} + δλ_k

        apply.

        M. A. Crisfield: A fast incremental/iterative solution procedure that handles "snap-through"
        Computers & Structures, 13:1-3, 55-62, 1981. https://doi.org/10.1016/0045-7949(81)90108-5
        """
        # shorthands to states from continuation object, for convenience
        Δx, dx = continuation.Δx, continuation.dx
        Δλ, dλ = continuation.Δλ.value.item(), continuation.dλ.value.item()
        dFdλ, δx_dFdλ = continuation.dFdλ, continuation.δx_dFdλ
        psi = continuation.psi
        ds = continuation.ds

        # pointers to states from snes object
        δx = snes.getSolutionUpdate()

        # clear existing solution update at first iteration
        if snes_iteration == 0:
            with δx.localForm() as δx_local:
                δx_local.set(0.0)

        # update accumulated increment
        Δx -= δx

        # compute δx_dFdλ if KSP is present and intact
        if snes.getKSP().getConvergedReason() != 0 and snes_iteration > 0:
            snes.getKSP().solve(-dFdλ, δx_dFdλ)
        else:
            with δx_dFdλ.localForm() as δx_dFdλ_local:
                δx_dFdλ_local.set(0.0)

        # intermediate value
        dFdλ_inner = psi**2 * continuation.inner(dFdλ, dFdλ)

        # coefficients of the quadratic equation: a1 * δλ**2 + 2 * a2 * δλ + a3 = 0
        a1 = continuation.inner(δx_dFdλ, δx_dFdλ) + dFdλ_inner
        a2 = continuation.inner(Δx, δx_dFdλ) + Δλ * dFdλ_inner
        a3 = continuation.inner(Δx, Δx) + Δλ**2 * dFdλ_inner - ds**2

        if a1 > 0.0:
            arg = a2**2 - a1 * a3

            if arg < 0.0:
                # something is seriously wrong: tangent does not intersect ball (ds)
                pprint(f"a1 = {a1:1.3e}, a2 = {a2:1.3e}, a3 = {a3:1.3e}")
                pprint(f"sqrt_arg = {arg:1.3e}")
                raise RuntimeError(
                    "Continuation: Failed solving the arc-length equation! Reduce ds."
                )

            sqr = arg ** (0.5)
            δλ1 = (-a2 - sqr) / a1
            δλ2 = (-a2 + sqr) / a1
            # pprint(f"δλ_1 = {δλ1:1.3e}, δλ_2 = {δλ2:1.3e}")

            sign = lambda x: bool(x > 0) - bool(x < 0)  # noqa: E731
            sign = sign(δx_dFdλ.dot(dx) + dλ * dFdλ_inner)

            δλ = δλ1 if δλ1 * sign > δλ2 * sign else δλ2
        else:
            δλ = -a3 / (2 * a2) if abs(a2) > 0.0 else 0.0

        # dolfiny.utils.pprint(f"δλ = {δλ:1.3e}")

        # update Δλ
        continuation.Δλ.value += δλ
        # update Δx
        continuation.Δx.axpy(δλ, δx_dFdλ)

        # update λ
        continuation.λ.value += δλ
        # update x
        snes.getSolution().axpy(δλ, δx_dFdλ)

        # compute residuals for updated (x, λ)
        λ_save = continuation.λ.value.copy()  # store λ value
        continuation.λ.value = 1.0  # evaluate F_1 = F(x, λ=1)
        snes.computeFunction(continuation.problem.active_x, continuation.dFdλ)
        continuation.λ.value = 0.0  # evaluate F_0 = F(x, λ=0)
        snes.computeFunction(continuation.problem.active_x, continuation.problem.active_F)
        with (
            continuation.dFdλ.localForm() as dFdλ_local,
            continuation.problem.active_F.localForm() as F_local,
        ):
            dFdλ_local -= F_local  # dFdλ = F_1 - F_0
        continuation.λ.value = λ_save  # store λ value
        snes.computeFunction(continuation.problem.active_x, continuation.problem.active_F)

        # monitor (default)
        x, dx, r = (
            abs(continuation.λ.value),
            abs(δλ),
            abs(continuation.inner(Δx, Δx) + Δλ**2 * dFdλ_inner - ds**2),
        )
        name = "λ"
        message = f"# arc           |x|={x:9.3e} |dx|={dx:9.3e} |r|={r:9.3e} ({name:s})"
        pprint(message)

        # monitor (custom)
        if continuation.monitor is not None:
            continuation.monitor(continuation)

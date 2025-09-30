from logging import Logger

from petsc4py import PETSc

import numpy as np

import dolfiny.logging
from dolfiny.la import negative_part, positive_part


class MMA:
    """Method of Moving Asymptotes (MMA).

    Optimisation solver for general constrained problems based on separable convex
    approximations.

    The method is applicable to general constrained optimisation problems in the (PETSc) form

        min     f(x)
         x
        s.t.    x⁻ ≤ x ≤ x⁺
                g(x) = 0 (TODO: coming soon)
                h(x) ≥ 0

    References:
        1) https://doi.org/10.1002/nme.1620240207
        2) https://people.kth.se/~krille/mmagcmma.pdf
        3) https://comsolyar.com/wp-content/uploads/2020/03/gcmma.pdf

    """

    # TODO: MMA.view() outputs ksp and ls info

    _f: float  # objective value

    # primal variables
    _x: PETSc.Vec  # type: ignore

    # dual variables
    _λ: PETSc.Vec  # type: ignore
    _J_λ: PETSc.Vec  # type: ignore

    # aux. variables
    _r: float
    _p: PETSc.Vec  # type: ignore
    _q: PETSc.Vec  # type: ignore
    _P: PETSc.Mat  # type: ignore
    _Q: PETSc.Mat  # type: ignore
    _logger: Logger

    # parameters
    _albefa: float
    _move_limit: float
    _asymptote_init: float
    _asymptote_decrement: float
    _asymptote_increment: float
    _asymptote_min: float
    _asymptote_max: float
    _raai: float
    _theta: float

    def __init__(self):
        self._logger = dolfiny.logging.logger.getChild(__name__)
        self._logger.debug(f"{__name__}.__init__")

        # Initialize all LA objects to None
        self._λ = None
        self._J_λ = None
        self._p = None
        self._q = None
        self._P = None
        self._Q = None

    def create(self, tao: PETSc.TAO) -> None:  # type: ignore
        self._logger.debug(f"{__name__}.create")

        # Default subsolver
        self._subsolver = PETSc.TAO().create()  # type: ignore
        self._subsolver.setType(PETSc.TAO.Type.BQNLS)  # type: ignore

    def setFromOptions(self, tao):
        self._logger.debug(f"{__name__}.setFromOptions")

        opts = PETSc.Options()

        prefix = tao.getOptionsPrefix()
        if prefix is None:
            prefix = ""

        self._albefa = opts.getReal(f"{prefix}tao_mma_albefa", 0.1)
        self._move_limit = opts.getReal(f"{prefix}tao_mma_move_limit", 0.5)
        self._asymptote_init = opts.getReal(f"{prefix}tao_mma_asymptote_init", 0.5)
        self._asymptote_decrement = opts.getReal(f"{prefix}tao_mma_asymptote_decrement", 0.7)
        self._asymptote_increment = opts.getReal(f"{prefix}tao_mma_asymptote_increment", 1.2)
        self._asymptote_min = opts.getReal(f"{prefix}tao_mma_asymptote_min", 0.01)
        self._asymptote_max = opts.getReal(f"{prefix}tao_mma_asymptote_max", 10.0)
        self._raai = opts.getReal(f"{prefix}tao_mma_raai", 0.0)  # 1e-5
        self._theta = opts.getReal(f"{prefix}tao_mma_theta", 0.001)

        if not np.isclose(self._raai, 0.0):
            raise RuntimeError("raai 0 only supported.")

        if self._asymptote_min > 1.0:
            raise RuntimeError(f"Asymptote min. ({self._asymptote_min}) must be ≤ 1.")

        if self._asymptote_max < 1.0:
            raise RuntimeError(f"Asymptote max. ({self._asymptote_max}) must be ≥ 1.")

        self._subsolver.setOptionsPrefix(f"{prefix}tao_mma_subsolver_")
        self._subsolver.setFromOptions()

        if not self._subsolver.getType().startswith("b"):
            raise RuntimeError("MMA subsolver needs to be a bound constrained solver.")

    def x(self, λ, x):
        """Dual to primal map.

        Compute from dual state λ the primal state x.

               √(p + λᵀP) ⊙ L + √(q + λᵀQ) ⊙ U
        x(λ) = -------------------------------
                   √(p + λᵀP) + √(q + λᵀQ)

        + bounds projection.
        """
        # tmp_p = √(p + λᵀP)
        tmp_p = x.copy()
        self._P.multTransposeAdd(λ, self._p, tmp_p)
        tmp_p.sqrtabs()

        # tmp_q = √(q + λᵀQ)
        tmp_q = x.copy()
        self._Q.multTransposeAdd(λ, self._q, tmp_q)
        tmp_q.sqrtabs()

        # x(λ) = (tmp_pᵀL + tmp_qᵀU ) ⨸ (tmp_p + tmp_q)
        div_tmp = tmp_p + tmp_q

        tmp_p.pointwiseMult(tmp_p, self._L)
        tmp_q.pointwiseMult(tmp_q, self._U)

        tmp_p.copy(x)
        x += tmp_q
        x.pointwiseDivide(x, div_tmp)

        # bounds projection
        x.pointwiseMax(self._alpha, x)
        x.pointwiseMin(self._beta, x)

    def setUp(self, tao: PETSc.TAO) -> None:  # type: ignore
        self._logger.debug(f"{__name__}.setUp")

        self._objective = 0.0
        self._gradient = tao.getGradient()[0]

        # Dual problem is a bound-constrained optimisation problem of the form
        #
        #     min   W(λ)
        #      λ
        #     s.t.    λ ≥ 0
        #
        if (constraint := tao.getInequalityConstraints())[1] is not None:
            self._λ = constraint[0].copy()
            self._λ.assemble()
            self._J_λ = self._λ.copy()
            self._J_λ.assemble()

        def dual_objective_and_gradient(tao, λ, G) -> float:
            self._logger.debug(f"{__name__}.dual_objective_and_gradient")

            # x(λ)
            self.x(λ, self._x)

            # Umx = (U - x)⁻¹
            Umx = self._U - self._x
            Umx.reciprocal()

            # xmL = (x - L)⁻¹
            xmL = self._x - self._L
            xmL.reciprocal()

            # tmp_p = p + λᵀP
            tmp_p = self._p.copy()
            self._P.multTransposeAdd(λ, tmp_p, tmp_p)

            # tmp_q = q + λᵀQ
            tmp_q = self._q.copy()
            self._Q.multTransposeAdd(λ, tmp_q, tmp_q)

            # W(λ) = r + λᵀr_h + (p + λᵀP) ⊙ (U - x(λ))⁻¹ + (q + λᵀQ) ⊙ (x(λ) - L)⁻¹
            #      = r + λᵀr_h + tmp_p ⊙ Umx + tmp_q ⊙ xmL
            # Note: we have no b term here as in the original MMA paper (compare pg. 365 eq. 20) due
            #       to different problem form.
            W = self._r
            W += λ.dot(self._r_h)
            W += tmp_p.dot(Umx)
            W += tmp_q.dot(xmL)

            # ∇W(λ) = r_h + P (U - x)⁻¹ + Q (x - L)⁻¹
            self._r_h.copy(G)
            self._P.multAdd(Umx, G, G)
            self._Q.multAdd(xmL, G, G)

            # Flip for max. to min.
            G.scale(-1)
            return -W  # type: ignore

        if (constraint := tao.getInequalityConstraints())[1] is not None:
            self._subsolver.setSolution(self._λ)
            self._subsolver.setObjectiveGradient(dual_objective_and_gradient, self._J_λ)

            lb = self._λ.copy()
            lb.set(0.0)
            ub = self._λ.copy()
            ub.set(PETSc.INFINITY)  # type: ignore
            self._subsolver.setVariableBounds((lb, ub))

            self._subsolver.setUp()

    def solve(self, tao):
        """Follows TaoSolve_Python_default."""
        self._logger.debug(f"{__name__}.solve")

        # TAO 0-th iteration is a convergence check.
        self._x = tao.getSolution()

        self._f = tao.computeObjectiveGradient(self._x, self._gradient)
        tao.monitor(f=self._f)

        c, h_tuple = tao.getInequalityConstraints()
        h, h_args, h_kwargs = h_tuple if h_tuple else (None, None, None)

        J, _, Jh_tuple = tao.getJacobianInequality()
        Jh, Jh_args, Jh_kwargs = Jh_tuple if Jh_tuple else (None, None, None)

        # TODO: workaround - see https://gitlab.com/petsc/petsc/-/merge_requests/8618.
        gatol, _, _ = tao.getTolerances()
        if self._gradient.norm() <= gatol:
            tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONVERGED_GATOL)

        lb, ub = tao.getVariableBounds()

        x_range = ub - lb

        if np.any(np.isinf(x_range)):
            raise RuntimeError("MMA requires a bounded domain.")

        # Init of asymptotes and sub-problem bounds
        # TODO: this should be only assignment here, move creation out of solve()
        self._L = lb.copy()
        self._U = ub.copy()

        self._alpha = lb.copy()
        self._beta = ub.copy()

        # Init history storage
        x_m1 = self._x.copy()
        x_m1.set(0.0)
        x_m2 = self._x.copy()
        x_m2.set(0.0)

        xmL = None
        Umx = None

        for it in range(1, tao.getMaximumIterations()):
            if tao.reason:
                break

            self._logger.debug(f"{__name__}.solve iteration {it}")

            # Compute f(x), ∇f(x), h(x) and J_h(x)
            self._f = tao.computeObjectiveGradient(self._x, self._gradient)

            if h:
                h(tao, self._x, c, *h_args, **h_kwargs)
                Jh(tao, self._x, J, None, *Jh_args, **Jh_kwargs)

                # The implemented MMA formulation relies on a form where h(x) ≤ 0 holds (not
                # h(x) ≥ 0). To account for this change we sign flip the callback
                # results - interpreting the constraint as if it was in the h(x) ≥ 0.
                c.scale(-1)
                if J.getType() == PETSc.Mat.Type.TRANSPOSE:
                    # Note: this ensures we never introduce a lazy scaling but, have the right
                    #       scaling on the underlying matrix, which we need for the splits based on
                    #       sign for P/Q.
                    J.getTransposeMat().scale(-1)
                else:
                    J.scale(-1)

            # Compute MMA subproblem dependencies

            # Update moving asymptotes L/U
            factor = self._x.copy()
            factor.set(1.0)
            factor.assemble()
            if it < 3:
                # L = x - asymptote_init * x_range
                self._x.copy(self._L)
                self._L.axpy(-self._asymptote_init, x_range)

                # U = x + asymptote_init * x_range
                self._x.copy(self._U)
                self._U.axpy(self._asymptote_init, x_range)
            else:
                diff_12 = self._x - x_m1
                diff_23 = x_m1 - x_m2

                sign_change = np.sign(diff_12.getArray()) * np.sign(diff_23.getArray())
                factor.setArray(
                    self._asymptote_increment * (sign_change > 0)
                    + self._asymptote_decrement * (sign_change < 0)
                    + 1.0 * (sign_change == 0)
                )

                # Note: the computation of L/U is based on the offsets of the previous iterate x_m1
                #       to the previous L/U.

                # L = x - f ⊙ (x_m1 - L)
                xmLf = x_m1 - self._L
                xmLf.pointwiseMult(xmLf, factor)

                self._x.copy(self._L)
                self._L -= xmLf

                # L_min = x - asymptote_max * x_range
                L_min = self._x.copy()
                L_min.axpy(-self._asymptote_max, x_range)

                # L_max = x - asymptote_min * x_range
                L_max = self._x.copy()
                L_max.axpy(-self._asymptote_min, x_range)

                # Bound project L to [L_min, L_max]
                self._L.pointwiseMax(self._L, L_min)
                self._L.pointwiseMin(self._L, L_max)

                # U = x + f ⊙ (U - x_m1)
                Umxf = self._U - x_m1
                Umxf.pointwiseMult(Umxf, factor)

                self._U = self._x.copy()
                self._U += Umxf

                # U_min = x + asymptote_min * x_range
                U_min = self._x.copy()
                U_min.axpy(self._asymptote_min, x_range)

                # U_max = x + asymptote_max * x_range
                U_max = self._x.copy()
                U_max.axpy(self._asymptote_max, x_range)

                # Bound project U to [U_min, U_max]
                self._U.pointwiseMax(self._U, U_min)
                self._U.pointwiseMin(self._U, U_max)

            # Umx = U - x
            Umx = self._U - self._x

            # xmL = x - L
            xmL = self._x - self._L

            # alpha = L + albefa * (x - L)
            self._L.copy(self._alpha)
            self._alpha.axpy(self._albefa, xmL)

            # alpha = max (alpha, x - move_limit * x_range)
            tmp = self._x.copy()
            tmp.axpy(-self._move_limit, x_range)
            self._alpha.pointwiseMax(self._alpha, tmp)

            # alpha = max ( alpha, x_min )
            self._alpha.pointwiseMax(self._alpha, lb)

            # beta = U - albefa * (U - x)
            self._U.copy(self._beta)
            self._beta.axpy(-self._albefa, Umx)

            # beta = min (beta, x + move_limit * x_range)
            tmp = self._x.copy()
            tmp.axpy(self._move_limit, x_range)
            self._beta.pointwiseMin(self._beta, tmp)

            # beta = min ( beta, x_max )
            self._beta.pointwiseMin(self._beta, ub)

            if not np.all(self._L.getArray() < self._alpha.getArray()):
                raise RuntimeError("L < alpha not fulfilled.")

            if not np.all(self._beta.getArray() < self._U.getArray()):
                raise RuntimeError("beta < U not fulfilled.")

            if not np.all(self._alpha.getArray() <= self._beta.getArray()):
                raise RuntimeError("alpha <= beta not fulfilled.")

            zero = self._x.copy()
            zero.set(0.0)

            grad_p = self._gradient.copy()
            grad_p.pointwiseMax(grad_p, zero)

            grad_m = self._gradient.copy()
            grad_m.scale(-1)
            grad_m.pointwiseMax(grad_m, zero)

            self._p = Umx.copy()
            self._p.pointwiseMult(self._p, self._p)

            # p_factor = (1+theta) * grad_p + theta * grad_m + raai / x_range
            p_factor = x_range.copy()
            p_factor.reciprocal()
            p_factor.scale(self._raai)
            p_factor.axpy(1 + self._theta, grad_p)
            p_factor.axpy(self._theta, grad_m)

            self._p.pointwiseMult(self._p, p_factor)

            self._q = xmL.copy()
            self._q.pointwiseMult(self._q, self._q)

            # q_factor = (1+theta) * grad_m + theta * grad_p + raai / x_range
            q_factor = x_range.copy()
            q_factor.reciprocal()
            q_factor.scale(self._raai)
            q_factor.axpy(1 + self._theta, grad_m)
            q_factor.axpy(self._theta, grad_p)

            self._q.pointwiseMult(self._q, q_factor)

            Umx_recp = Umx.copy()
            Umx_recp.reciprocal()
            xmL_recp = xmL.copy()
            xmL_recp.reciprocal()
            self._r = self._f - Umx_recp.dot(self._p) - xmL_recp.dot(self._q)

            # Update history (before overwriting x with new solution)
            x_m1.copy(x_m2)
            self._x.copy(x_m1)

            if h:
                self._r_h = c.copy()  # h(x)

                J_p = J.copy()
                if J_p.getType() == PETSc.Mat.Type.TRANSPOSE:
                    positive_part(J_p.getTransposeMat())
                else:
                    positive_part(J_p)

                J_m = J.copy()
                if J_m.getType() == PETSc.Mat.Type.TRANSPOSE:
                    negative_part(J_m.getTransposeMat())
                else:
                    negative_part(J_m)

                # P = (1+theta) J_p + theta J_m + TODO figure kappa out
                self._P = J_p.copy()
                self._P.scale(1 + self._theta)
                self._P.axpy(self._theta, J_m)

                tmp = Umx.copy()
                tmp.pointwiseMult(tmp, tmp)

                self._P.diagonalScale(L=None, R=tmp)

                # r_h -= P (U - x)⁻¹
                tmp = self._r_h.copy()
                self._P.mult(Umx_recp, tmp)
                self._r_h -= tmp

                # Q = (1+theta) J_m + theta J_p + TODO figure kappa out
                self._Q = J_m.copy()
                self._Q.scale(1 + self._theta)
                self._Q.axpy(self._theta, J_p)

                tmp = xmL.copy()
                tmp.pointwiseMult(tmp, tmp)

                self._Q.diagonalScale(L=None, R=tmp)

                # r_h -= Q (x - L)⁻¹
                tmp = self._r_h.copy()
                self._Q.mult(xmL_recp, tmp)
                self._r_h -= tmp

                self._subsolver.solve()

                if self._subsolver.getConvergedReason() < 0:
                    raise RuntimeError("Subsolver diverged.")

                self.x(self._subsolver.getSolution(), self._x)
            else:
                tmp_p = self._p.copy()
                tmp_p.sqrtabs()

                tmp_q = self._q.copy()
                tmp_q.sqrtabs()

                div_tmp = tmp_p.copy()
                div_tmp += tmp_q

                tmp_p.pointwiseMult(tmp_p, self._L)
                tmp_q.pointwiseMult(tmp_q, self._U)

                tmp_p.copy(self._x)
                self._x += tmp_q
                self._x.pointwiseDivide(self._x, div_tmp)

                self._x.pointwiseMax(self._x, lb)
                self._x.pointwiseMin(self._x, ub)

            # convergence and logging
            self._objective = tao.computeObjectiveGradient(self._x, self._gradient)

            # TODO: workaround - see https://gitlab.com/petsc/petsc/-/merge_requests/8618.
            if self._gradient.norm() <= gatol:
                tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONVERGED_GATOL)

            tao.setIterationNumber(it)

            tao.monitor(f=self._objective)

    @property
    def subsolver(self) -> PETSc.TAO:  # type: ignore
        return self._subsolver

    @property
    def albefa(self) -> float:
        return self._albefa

    @property
    def move_limit(self) -> float:
        return self._move_limit

    @property
    def asymptote_init(self) -> float:
        return self._asymptote_init

    @property
    def asymptote_decrement(self) -> float:
        return self._asymptote_decrement

    @property
    def asymptote_increment(self) -> float:
        return self._asymptote_increment

    @property
    def asymptote_min(self) -> float:
        return self._asymptote_min

    @property
    def asymptote_max(self) -> float:
        return self._asymptote_max

    @property
    def raai(self) -> float:
        return self._raai

    @property
    def theta(self) -> float:
        return self._theta

    def destroy(self, tao: PETSc.TAO):  # type: ignore
        self._logger.debug(f"{__name__}.destroy")

        if self._λ is not None:
            self._λ.destroy()

        if self._J_λ is not None:
            self._J_λ.destroy()

        if self._p is not None:
            self._p.destroy()

        if self._q is not None:
            self._q.destroy()

        if self._P is not None:
            self._P.destroy()

        if self._Q is not None:
            self._Q.destroy()

        self._subsolver.destroy()

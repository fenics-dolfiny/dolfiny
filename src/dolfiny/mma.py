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

        self._x_range = None
        self._x_m1 = None
        self._x_m2 = None
        self._diff_12 = None
        self._diff_23 = None

        self._p = None
        self._q = None

        self._L = None
        self._U = None
        self._alpha = None
        self._beta = None

        self._xmL = None
        self._Umx = None
        self._xmL_recp = None
        self._Umx_recp = None

        self._tmp = None
        self._tmp_2 = None
        self._tmp_3 = None

        self._zero = None

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
        # tmp = √(p + λᵀP)
        self._P.multTransposeAdd(λ, self._p, self._tmp)
        self._tmp.sqrtabs()

        # tmp_2 = √(q + λᵀQ)
        self._Q.multTransposeAdd(λ, self._q, self._tmp_2)
        self._tmp_2.sqrtabs()

        # x(λ) = (tmpᵀL + tmp_2ᵀU ) ⨸ (tmp + tmp_2)
        self._tmp.copy(self._tmp_3)
        self._tmp_3 += self._tmp_2

        self._tmp.pointwiseMult(self._tmp, self._L)
        self._tmp_2.pointwiseMult(self._tmp_2, self._U)

        self._tmp.copy(x)
        x += self._tmp_2
        x.pointwiseDivide(x, self._tmp_3)

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

            # tmp_2 = (U - x)⁻¹
            self._U.copy(self._tmp_2)
            self._tmp_2 -= self._x
            self._tmp_2.reciprocal()

            # tmp_3 = (x - L)⁻¹
            self._x.copy(self._tmp_3)
            self._tmp_3 -= self._L
            self._tmp_3.reciprocal()

            # W(λ) = r + λᵀr_h + (p + λᵀP) ⊙ (U - x(λ))⁻¹ + (q + λᵀQ) ⊙ (x(λ) - L)⁻¹
            #      = r + λᵀr_h + tmp_p ⊙ Umx + tmp_q ⊙ xmL
            # Note: we have no b term here as in the original MMA paper (compare pg. 365 eq. 20) due
            #       to different problem form.
            W = self._r
            W += λ.dot(self._r_h)

            # tmp = p + λᵀP
            self._P.multTransposeAdd(λ, self._p, self._tmp)
            W += self._tmp.dot(self._tmp_2)

            # tmp_q = q + λᵀQ
            self._Q.multTransposeAdd(λ, self._q, self._tmp)
            W += self._tmp.dot(self._tmp_3)

            # ∇W(λ) = r_h + P (U - x)⁻¹ + Q (x - L)⁻¹
            self._r_h.copy(G)
            self._P.multAdd(self._tmp_2, G, G)
            self._Q.multAdd(self._tmp_3, G, G)

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

        # variables/intermediates
        self._x = tao.getSolution()

        self._x_range = self._x.copy()
        self._x_m1 = self._x.copy()
        self._x_m2 = self._x.copy()
        self._diff_12 = self._x.copy()
        self._diff_23 = self._x.copy()

        self._p = self._x.copy()
        self._q = self._x.copy()

        self._L = self._x.copy()
        self._U = self._x.copy()
        self._alpha = self._x.copy()
        self._beta = self._x.copy()

        self._xmL = self._x.copy()
        self._Umx = self._x.copy()
        self._xmL_recp = self._x.copy()
        self._Umx_recp = self._x.copy()

        self._tmp = self._x.copy()
        self._tmp_2 = self._x.copy()
        self._tmp_3 = self._x.copy()

        self._zero = self._x.copy()
        self._zero.set(0.0)

    def solve(self, tao):
        """Follows TaoSolve_Python_default."""
        self._logger.debug(f"{__name__}.solve")

        # TAO 0-th iteration is a convergence check.

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

        # x_range = ub - lb
        ub.copy(self._x_range)
        self._x_range -= lb

        if np.any(np.isinf(self._x_range)):
            raise RuntimeError("MMA requires a bounded domain.")

        # Initial lower/upper bounds
        lb.copy(self._L)
        ub.copy(self._U)
        lb.copy(self._alpha)
        ub.copy(self._beta)

        # Reset history
        self._x_m1.set(0.0)
        self._x_m2.set(0.0)

        if c:
            tmp_h = c.copy()
            self._r_h = c.copy()

        if J:
            J_p = J.copy()
            J_m = J.copy()
            self._P = J.copy()
            self._Q = J.copy()

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
            self._tmp_2.set(1.0)
            if it < 3:
                # L = x - asymptote_init * x_range
                self._x.copy(self._L)
                self._L.axpy(-self._asymptote_init, self._x_range)

                # U = x + asymptote_init * x_range
                self._x.copy(self._U)
                self._U.axpy(self._asymptote_init, self._x_range)
            else:
                self._x.copy(self._diff_12)
                self._diff_12 -= self._x_m1

                self._x_m1.copy(self._diff_23)
                self._diff_23 -= self._x_m2

                sign_change = np.sign(self._diff_12.getArray()) * np.sign(self._diff_23.getArray())
                self._tmp_2.setArray(
                    self._asymptote_increment * (sign_change > 0)
                    + self._asymptote_decrement * (sign_change < 0)
                    + 1.0 * (sign_change == 0)
                )

                # Note: the computation of L/U is based on the offsets of the previous iterate x_m1
                #       to the previous L/U.

                # L = x - f ⊙ (x_m1 - L)
                self._x_m1.copy(self._tmp)
                self._tmp -= self._L
                self._tmp.pointwiseMult(self._tmp, self._tmp_2)

                self._x.copy(self._L)
                self._L -= self._tmp

                # Bound project L to [L_min, L_max]

                # tmp = x - asymptote_max * x_range
                self._x.copy(self._tmp)
                self._tmp.axpy(-self._asymptote_max, self._x_range)

                self._L.pointwiseMax(self._L, self._tmp)

                # tmp = x - asymptote_min * x_range
                self._x.copy(self._tmp)
                self._tmp.axpy(-self._asymptote_min, self._x_range)

                self._L.pointwiseMin(self._L, self._tmp)

                # U = x + f ⊙ (U - x_m1)
                self._U.copy(self._tmp)
                self._tmp -= self._x_m1
                self._tmp.pointwiseMult(self._tmp, self._tmp_2)

                self._x.copy(self._U)
                self._U += self._tmp

                # Bound project U to [U_min, U_max]
                # U_min = x + asymptote_min * x_range
                self._x.copy(self._tmp)
                self._tmp.axpy(self._asymptote_min, self._x_range)

                self._U.pointwiseMax(self._U, self._tmp)

                # U_max = x + asymptote_max * x_range
                self._x.copy(self._tmp)
                self._tmp.axpy(self._asymptote_max, self._x_range)

                self._U.pointwiseMin(self._U, self._tmp)

            # Umx = U - x
            self._U.copy(self._Umx)
            self._Umx -= self._x

            # xmL = x - L
            self._x.copy(self._xmL)
            self._xmL -= self._L

            # alpha = L + albefa * (x - L)
            self._L.copy(self._alpha)
            self._alpha.axpy(self._albefa, self._xmL)

            # alpha = max (alpha, x - move_limit * x_range)
            self._x.copy(self._tmp)
            self._tmp.axpy(-self._move_limit, self._x_range)
            self._alpha.pointwiseMax(self._alpha, self._tmp)

            # alpha = max ( alpha, x_min )
            self._alpha.pointwiseMax(self._alpha, lb)

            # beta = U - albefa * (U - x)
            self._U.copy(self._beta)
            self._beta.axpy(-self._albefa, self._Umx)

            # beta = min (beta, x + move_limit * x_range)
            self._x.copy(self._tmp)
            self._tmp.axpy(self._move_limit, self._x_range)
            self._beta.pointwiseMin(self._beta, self._tmp)

            # beta = min ( beta, x_max )
            self._beta.pointwiseMin(self._beta, ub)

            if not np.all(self._L.getArray() < self._alpha.getArray()):
                raise RuntimeError("L < alpha not fulfilled.")

            if not np.all(self._beta.getArray() < self._U.getArray()):
                raise RuntimeError("beta < U not fulfilled.")

            if not np.all(self._alpha.getArray() <= self._beta.getArray()):
                raise RuntimeError("alpha <= beta not fulfilled.")

            # tmp_2 = grad_p
            self._gradient.copy(self._tmp_2)
            self._tmp_2.pointwiseMax(self._tmp_2, self._zero)

            # tmp_3 = grad_m
            self._gradient.copy(self._tmp_3)
            self._tmp_3.scale(-1)
            self._tmp_3.pointwiseMax(self._tmp_3, self._zero)

            self._Umx.copy(self._p)
            self._p.pointwiseMult(self._p, self._p)

            # tmp = (1+theta) * grad_p + theta * grad_m + raai / x_range
            self._x_range.copy(self._tmp)
            self._tmp.reciprocal()
            self._tmp.scale(self._raai)
            self._tmp.axpy(1 + self._theta, self._tmp_2)
            self._tmp.axpy(self._theta, self._tmp_3)

            self._p.pointwiseMult(self._p, self._tmp)

            self._xmL.copy(self._q)
            self._q.pointwiseMult(self._q, self._q)

            # tmp = (1+theta) * grad_m + theta * grad_p + raai / x_range
            self._x_range.copy(self._tmp)
            self._tmp.reciprocal()
            self._tmp.scale(self._raai)
            self._tmp.axpy(1 + self._theta, self._tmp_3)
            self._tmp.axpy(self._theta, self._tmp_2)

            self._q.pointwiseMult(self._q, self._tmp)

            self._Umx.copy(self._Umx_recp)
            self._Umx_recp.reciprocal()
            self._xmL.copy(self._xmL_recp)
            self._xmL_recp.reciprocal()
            self._r = self._f - self._Umx_recp.dot(self._p) - self._xmL_recp.dot(self._q)

            # Update history (before overwriting x with new solution)
            self._x_m1.copy(self._x_m2)
            self._x.copy(self._x_m1)

            if h:
                c.copy(self._r_h)

                J.copy(J_p)
                if J_p.getType() == PETSc.Mat.Type.TRANSPOSE:
                    positive_part(J_p.getTransposeMat())
                else:
                    positive_part(J_p)

                J.copy(J_m)
                if J_m.getType() == PETSc.Mat.Type.TRANSPOSE:
                    negative_part(J_m.getTransposeMat())
                else:
                    negative_part(J_m)

                # P = (1+theta) J_p + theta J_m + TODO figure kappa out
                J_p.copy(self._P)
                self._P.scale(1 + self._theta)
                self._P.axpy(self._theta, J_m)

                self._Umx.copy(self._tmp)
                self._tmp.pointwiseMult(self._tmp, self._tmp)

                self._P.diagonalScale(L=None, R=self._tmp)

                # r_h -= P (U - x)⁻¹
                self._P.mult(self._Umx_recp, tmp_h)
                self._r_h -= tmp_h

                # Q = (1+theta) J_m + theta J_p + TODO figure kappa out
                J_m.copy(self._Q)
                self._Q.scale(1 + self._theta)
                self._Q.axpy(self._theta, J_p)

                self._xmL.copy(self._tmp)
                self._tmp.pointwiseMult(self._tmp, self._tmp)

                self._Q.diagonalScale(L=None, R=self._tmp)

                # r_h -= Q (x - L)⁻¹
                self._Q.mult(self._xmL_recp, tmp_h)
                self._r_h -= tmp_h

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

            tao.setIterationNumber(it)
            tao.monitor(f=self._objective, res=self._gradient.norm())  # TODO: cnorm
            tao.checkConverged()

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

        to_destroy = (
            self._λ,
            self._J_λ,
            self._x_range,
            self._x_m1,
            self._x_m2,
            self._diff_12,
            self._diff_23,
            self._p,
            self._q,
            self._P,
            self._Q,
            self._L,
            self._U,
            self._alpha,
            self._beta,
            self._xmL,
            self._Umx,
            self._xmL_recp,
            self._Umx_recp,
            self._tmp,
            self._tmp_2,
            self._tmp_3,
            self._zero,
        )
        for o in filter(lambda o: o is not None, to_destroy):
            o.destroy()

        self._subsolver.destroy()

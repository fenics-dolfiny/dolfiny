from logging import Logger

from petsc4py import PETSc

import dolfiny.logging
from dolfiny.la import negative_part, positive_part


class CONLIN:
    """Convex linearization method (CONLIN).

    Optimisation solver for general constrained problems based on separable convex
    approximations.

    The method is applicable to general constrained optimisation problems in the (PETSc) form

        min     f(x)
         x
        s.t.    x⁻ ≤ x ≤ x⁺
                g(x) = 0 (TODO: coming soon)
                h(x) ≥ 0

    References:
        1) https://doi.org/10.1007/BF01637664

    """

    # TODO: CONLIN.view() outputs ksp and ls info

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

    # options
    _grad_eps: float

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

        prefix = tao.getOptionsPrefix()
        if prefix is None:
            prefix = ""

        opts = PETSc.Options()
        self._grad_eps = opts.getReal(f"{prefix}tao_conlin_grad_eps", 1e-6)

        self._subsolver.setOptionsPrefix(f"{prefix}tao_conlin_subsolver_")
        self._subsolver.setFromOptions()

        if not self._subsolver.getType().startswith("b"):
            raise RuntimeError("CONLIN subsolver needs to be a bound constrained solver.")

    def x(self, λ, x):
        """Dual to primal map.

        Compute from dual state λ the primal state x.

               √(q + λᵀQ)
        x(λ) = ----------
               √(p + λᵀP)

        + bounds projection.
        """
        # tmp = √(p + λᵀP)
        self._P.multTransposeAdd(λ, self._p, self._tmp)
        self._tmp.sqrtabs()

        # x(λ) = √(q + λᵀQ) ⨸ tmp
        self._Q.multTransposeAdd(λ, self._q, x)
        x.sqrtabs()

        x.pointwiseDivide(x, self._tmp)

        # bounds projection
        x.pointwiseMax(self._lb, x)
        x.pointwiseMin(self._ub, x)

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

            self._x.copy(self._x_recip)
            self._x_recip.reciprocal()

            # W(λ) = r + λᵀr_h + (p + λᵀP) x(λ) + (q + λᵀQ) ⊙ x(λ)⁻¹
            #      = r + λᵀr_h + tmp_p ⊙ x(λ) + tmp_q ⊙ x(λ)⁻¹
            W = self._r
            W += λ.dot(self._r_h)

            # tmp = p + λᵀP
            self._P.multTransposeAdd(λ, self._p, self._tmp)

            W += self._tmp.dot(self._x)

            # tmp_q = q + λᵀQ
            self._Q.multTransposeAdd(λ, self._q, self._tmp)

            W += self._tmp.dot(self._x_recip)

            # ∇W(λ) = r_h + P x + Q x⁻¹
            self._r_h.copy(G)
            self._P.multAdd(self._x, G, G)
            self._Q.multAdd(self._x_recip, G, G)

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

        self._x = tao.getSolution()

        self._p = self._x.copy()
        self._q = self._x.copy()
        self._x_recip = self._x.copy()

        self._zero = self._x.copy()
        self._zero.set(self._grad_eps)

        self._tmp = self._x.copy()

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

        self._lb, self._ub = tao.getVariableBounds()

        if c:
            tmp_h = c.copy()
            self._r_h = c.copy()

        if J:
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

                # The implemented CONLIN formulation relies on a form where h(x) ≤ 0 holds (not
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

            # p = ∇f(x₀)⁺
            self._gradient.copy(self._p)
            self._p.pointwiseMax(self._p, self._zero)

            # q = ∇f(x₀)⁻ ⊙ x₀⁻²
            self._gradient.copy(self._q)
            self._q.scale(-1)
            self._q.pointwiseMax(self._q, self._zero)

            self._x.copy(self._tmp)
            # x0_square = self._x.copy()  # TODO
            self._tmp.pointwiseMult(self._tmp, self._tmp)
            self._q.pointwiseMult(self._q, self._tmp)

            self._x.copy(self._x_recip)
            self._x_recip.reciprocal()
            self._r = self._f - self._p.dot(self._x) - self._q.dot(self._x_recip)

            if h:
                c.copy(self._r_h)

                J.copy(self._P)
                if self._P.getType() == PETSc.Mat.Type.TRANSPOSE:
                    positive_part(self._P.getTransposeMat())
                else:
                    positive_part(self._P)

                J.copy(self._Q)
                if self._Q.getType() == PETSc.Mat.Type.TRANSPOSE:
                    negative_part(self._Q.getTransposeMat())
                else:
                    negative_part(self._Q)

                self._Q.diagonalScale(L=None, R=self._tmp)

                # r_h -= P x
                self._P.mult(self._x, tmp_h)
                self._r_h -= tmp_h

                # r_h -= Q x⁻¹
                self._Q.mult(self._x_recip, tmp_h)
                self._r_h -= tmp_h

                self._subsolver.solve()

                if self._subsolver.getConvergedReason() < 0:
                    raise RuntimeError("Subsolver diverged.")

                self.x(self._subsolver.getSolution(), self._x)
            else:
                # x = √(q / p)
                self._x.pointwiseDivide(self._q, self._p)
                self._x.sqrtabs()

                self._x.pointwiseMax(self._x, self._lb)
                self._x.pointwiseMin(self._x, self._ub)

            # convergence and logging
            self._objective = tao.computeObjectiveGradient(self._x, self._gradient)

            # TODO: workaround - see https://gitlab.com/petsc/petsc/-/merge_requests/8618.
            if self._gradient.norm() <= gatol:
                tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONVERGED_GATOL)

            tao.setIterationNumber(it)
            tao.monitor(f=self._objective, res=self._gradient.norm())  # TODO: cnorm
            tao.checkConverged()

        if tmp_h is not None:
            tmp_h.destroy()

    @property
    def subsolver(self) -> PETSc.TAO:  # type: ignore
        return self._subsolver

    def destroy(self, tao: PETSc.TAO):  # type: ignore
        self._logger.debug(f"{__name__}.destroy")

        to_destroy = (
            self._λ,
            self._J_λ,
            self._p,
            self._q,
            self._P,
            self._Q,
            self._zero,
            self._r_h,
            self._x_recip,
        )
        for o in filter(lambda o: o is not None, to_destroy):
            o.destroy()

        self._subsolver.destroy()

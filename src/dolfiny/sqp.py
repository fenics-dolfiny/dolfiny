from logging import Logger

from petsc4py import PETSc

import dolfiny.logging


class SQP:
    """Sequential quadratic optimisation.

    Optimisation solver for general constrained problems based on quadratic approximations.

    The method is applicable to general constrained optimisation problems in the (PETSc) form

        min     f(x)
         x
        s.t.    x⁻ ≤ x ≤ x⁺
                g(x) = 0
                h(x) ≥ 0 (TODO: coming soon)

    References:
        1) https://doi.org/10.1017/9781108980647

    For every iteration at current iterate x̂, the optimisation is approximated by a quadratic
    auxiliary problem, given by

        min     f̂(x)
         x
        s.t.    x⁻ ≤ x ≤ x⁺
                ĝ(x) = 0
                ĥ(x) ≥ 0

    where

        f̂(x) = f(x̂) + ∇f(x̂)^T (x - x̂) + 1/2 (x - x̂)^T Hf(x̂) (x - x̂)
        ĝ(x) = g(x̂) + ∇g(x̂)^T (x - x̂)
        ĥ(x) = h(x̂) + ∇h(x̂)^T (x - x̂)

    Note:
        ⇒ ∇f̂(x) = ∇f(x̂) + Hf(x̂) (x - x̂)
        ⇒ Hf̂(x) = Hf(x̂).

        ⇒ ∇ĝ(x) = ∇g(x̂)
        ⇒ Hĝ(x) = 0

        ⇒ ∇ĥ(x) = ∇h(x̂)
        ⇒ Hĥ(x) = 0

    The subsolver takes PETSc options marked with the prefix 'sqp_subsolver_'.

    """

    _objective: float
    _logger: Logger

    def __init__(self) -> None:
        self._logger = dolfiny.logging.logger.getChild(__name__)
        self._logger.debug(f"{__name__}.__init__")

        self._constraint = None
        self._constraint_jacobian = None

    def setUp(self, tao: PETSc.TAO) -> None:  # type: ignore
        self._logger.debug(f"{__name__}.setUp")

    def create(self, tao: PETSc.TAO) -> None:  # type: ignore
        self._logger.debug(f"{__name__}.create")

        self._subsolver = PETSc.KSP().create()  # type: ignore
        self._subsolver.setOptionsPrefix("sqp_subsolver_")

    def setFromOptions(self, tao: PETSc.TAO) -> None:  # type: ignore
        self._logger.debug(f"{__name__}.setFromOptions")

        prefix = tao.getOptionsPrefix()
        if prefix is None:
            prefix = ""

        self._subsolver.setOptionsPrefix(f"{prefix}sqp_subsolver_")
        self._subsolver.setFromOptions()

    def solve(self, tao: PETSc.TAO) -> None:  # type: ignore
        """Follows TaoSolve_Python_default."""
        self._logger.debug(f"{__name__}.solve")

        # TAO 0-th iteration is a convergence check (only).
        x = tao.getSolution()
        grad = tao.getGradient()[0]
        H = tao.getHessian()[0]
        self._objective = tao.computeObjectiveGradient(x, grad)

        c, g_tuple = tao.getEqualityConstraints()
        g, g_args, g_kwargs = g_tuple if g_tuple else (None, None, None)

        J, _, Jg_tuple = tao.getJacobianEquality()
        Jg, Jg_args, Jg_kwargs = Jg_tuple if Jg_tuple else (None, None, None)

        tao.setIterationNumber(0)
        tao.monitor(its=0, f=self._objective, res=grad.norm())  # TODO: norm for constrained case
        tao.checkConverged()

        lb, ub = tao.getVariableBounds()

        for it in range(1, tao.getMaximumIterations()):
            if tao.reason:
                break

            self._logger.debug(f"{__name__}.solve iteration {it}")

            tao.computeHessian(x, H)

            if not g:
                # In the unconstrained case we have:
                #      0 = ∇f̂(x) = ∇f(x̂) + Hf(x̂) (x - x̂)
                # ⟺   Hf(x̂) x = Hf(x̂) x̂ - ∇f(x̂)
                #
                # ⇒ A = Hf(x̂), b = Hf(x̂) x̂ - ∇f(x̂)
                self._subsolver.setOperators(H)
                b = x.copy()
                H.mult(x, b)
                b -= grad
                self._subsolver.solve(b, x)
            else:
                # In the eq. constrained case we have:
                #
                #   L(x, λ) = f̂(x) + λᵀĥ(x)
                #           = f(x̂) + ∇f(x̂)ᵀ (x - x̂) + 1/2 (x - x̂)ᵀ Hf(x̂) (x - x̂)
                #             + λᵀ(h(x̂) + ∇h(x̂)ᵀ (x - x̂))
                #
                # KKT:
                #      0 = Lₓ (x, λ) = ∇f̂(x) + λᵀ∇ĥ(x)
                #                    = ∇f(x̂) + Hf(x̂) (x - x̂) + λᵀ∇h(x̂)
                #      0 = Lλ (x, λ) = ĥ(x)
                #                    = h(x̂) + ∇h(x̂)ᵀ (x - x̂)
                #
                # ⟺  ⎡ Hf(x̂) ∇h(x̂)ᵀ ⎤ ⎡ x ⎤ = ⎡ Hf(x̂) x̂ - ∇f(x̂) ⎤
                #     ⎣ ∇h(x̂)    0   ⎦ ⎣ λ ⎦   ⎣ ∇h(x̂) x̂ - h(x̂)  ⎦
                #
                # ⇒ A = ⎡ Hf(x̂) ∇h(x̂)ᵀ ⎤, b = ⎡ Hf(x̂) x̂ - ∇f(x̂) ⎤
                #       ⎣ ∇h(x̂)    0   ⎦      ⎣ ∇h(x̂) x̂ - h(x̂)  ⎦
                assert Jg

                g(tao, x, c, *g_args, **g_kwargs)
                Jg(tao, x, J, None, *Jg_args, **Jg_kwargs)

                A = PETSc.Mat().createNest([[H, PETSc.Mat().createTranspose(J)], [J, None]])  # type: ignore
                self._subsolver.setOperators(A)

                b_x = x.copy()
                H.mult(x, b_x)
                b_x -= grad

                b_λ = c.copy()
                J.mult(x, b_λ)
                b_λ -= c

                b = PETSc.Vec().createNest([b_x, b_λ])  # type: ignore
                λ = c.copy()
                λ.set(0.0)
                block_x = PETSc.Vec().createNest([x, λ])  # type: ignore
                self._subsolver.solve(b, block_x)

            if lb:
                x.pointwiseMax(lb, x)

            if ub:
                x.pointwiseMin(ub, x)

            self._objective = tao.computeObjectiveGradient(x, grad)

            tao.setIterationNumber(it)
            tao.monitor(
                its=it, f=self._objective, res=grad.norm()
            )  # TODO: norm for constrained case
            tao.checkConverged()

    def getObjectiveValue(self) -> float:
        return self._objective

    @property
    def subsolver(self) -> PETSc.KSP:  # type: ignore
        return self._subsolver

    def destroy(self, tao: PETSc.TAO) -> None:  # type: ignore
        self._logger.debug(f"{__name__}.destroy")

        self._subsolver.destroy()

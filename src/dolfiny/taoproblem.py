from collections.abc import Sequence

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.typing import (
    TAOConstraintsFunction,
    TAOConstraintsJacobianFunction,
    TAOHessianFunction,
    TAOJacobianFunction,
    TAOObjectiveFunction,
)

import dolfinx
import dolfinx.fem.petsc
import dolfinx.la.petsc
import ufl
import ufl.equation

import numpy as np

import dolfiny
import dolfiny.inequality
from dolfiny.utils import ANSI, attributes_to_dict, pprint, prefixify


def sync_functions(u: Sequence[dolfinx.fem.Function]):
    """Create wrapper that synchronizes given functions.

    Parameters
    ----------
    u:
        Functions that the returned wrapped will update on invocation.

    Returns
    -------
    Decorator for TAO callbacks that wraps callback invocations with vector to function
    assignment.

    """

    def _decorator(_to_wrap):
        def _wrapped_callback(tao: PETSc.TAO, x: PETSc.Vec, *args) -> float | None:  # type: ignore
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
            dolfinx.fem.petsc.assign(x, u)

            return _to_wrap(tao, x, *args)  # type: ignore

        return _wrapped_callback

    return _decorator


def wrap_objective_callbacks(
    comm: MPI.Comm,
    u: Sequence[dolfinx.fem.Function],
    F: ufl.Form,
    J: Sequence[ufl.Form] | None = None,
    H: Sequence[Sequence[ufl.Form | None]] | None = None,
    bcs: Sequence[dolfinx.fem.DirichletBC] = [],
    form_compiler_options: dict[str, str] | None = None,
    jit_options: dict[str, str] | None = None,
) -> tuple[  # type: ignore
    PETSc.Vec,
    TAOObjectiveFunction,
    tuple[TAOJacobianFunction, PETSc.Vec],
    tuple[TAOHessianFunction, PETSc.Mat],
]:
    """Create objective, Jacobian and Hessian routines from form.

    Note:
        Boundary conditions are enforced through the Jacobian, consistent with DOLFINx.
        This implies that boundary conditions are not enforced on object evaluation - a gradient
        bases decent is necessary to incorporate those correctly.

    Parameters
    ----------
    comm:
        Communicator that all functions have in common.
    u:
        The list of arguments the form is defined over.
    F:
        Objective form.
    J:
        Jacobian forms.
    H:
        Hessian forms.
    bcs:
        List of boundary conditions that the callbacks will apply.
    form_compiler_options:
        Option for form compilation, i.e. FFCx options.
    jit_options:
        Compiler flags to use during form compilation.

    Returns
    -------
    Tuple of x-vector associated with the form F, wrapped evaluation callback, tuple of Jacobian
    vector and wrapped evaluation callback, and tuple of Hessian matrix and wrapped evaluation
    callback.

    """
    if J is None:
        δu = ufl.TestFunctions(ufl.MixedFunctionSpace(*(_u.function_space for _u in u)))
        J = ufl.derivative(F, u, δu)
        J = ufl.extract_blocks(J)

    if H is None:
        assert J is not None

        H = [[ufl.derivative(_J, _u, ufl.TrialFunction(_u.function_space)) for _u in u] for _J in J]
        H = [[None if (e is None or e.empty()) else e for e in row] for row in H]

    def compile(form):
        return dolfinx.fem.form(
            form, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

    F_form: dolfinx.fem.Form = compile(F)
    J_form: Sequence[dolfinx.fem.Form] = compile(J)
    H_form: Sequence[Sequence[dolfinx.fem.Form]] = compile(H)

    x0 = dolfinx.fem.petsc.create_vector(J_form, kind=PETSc.Vec.Type.MPI)  # type: ignore

    jacobian = x0.copy()
    jacobian.setAttr("_blocks", x0.getAttr("_blocks"))

    hessian = dolfinx.fem.petsc.create_matrix(H_form)

    @sync_functions(u)
    def _callback_F(tao: PETSc.TAO, x: PETSc.Vec) -> float:  # type: ignore
        F_value = dolfinx.fem.assemble_scalar(F_form)
        assert isinstance(F_value, float)
        return comm.allreduce(F_value, op=MPI.SUM)  # type: ignore

    @sync_functions(u)
    def _callback_J(tao: PETSc.TAO, x: PETSc.Vec, J_vec: PETSc.Vec) -> None:  # type: ignore
        x.setAttr("_blocks", x0.getAttr("_blocks"))
        J_vec.setAttr("_blocks", x0.getAttr("_blocks"))
        J_vec.zeroEntries()
        J_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

        dolfinx.fem.petsc.assemble_vector(J_vec, J_form)  # type: ignore

        dolfinx.fem.petsc.apply_lifting(
            J_vec,
            a=H_form,
            bcs=dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(H_form, 1), bcs),
            x0=x,
            alpha=-1.0,
        )

        J_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore

        dolfinx.fem.petsc.set_bc(
            J_vec,
            bcs=dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(J_form), bcs),
            x0=x,
            alpha=-1.0,
        )

    @sync_functions(u)
    def _callback_H(tao: PETSc.TAO, x: PETSc.Vec, H_mat: PETSc.Mat, P_mat: PETSc.Mat) -> None:  # type: ignore
        x.setAttr("_blocks", x0.getAttr("_blocks"))
        H_mat.zeroEntries()

        dolfinx.fem.petsc.assemble_matrix(H_mat, a=H_form, bcs=bcs, diag=1.0)  # type: ignore

        H_mat.assemble()

        if P_mat != H:
            H_mat.copy(P_mat)
            P_mat.assemble()

    return x0, _callback_F, (_callback_J, jacobian), (_callback_H, hessian)


def wrap_constraint_callbacks(
    comm: MPI.Comm,
    x0: PETSc.Vec,  # type: ignore
    u: Sequence[dolfinx.fem.Function],
    g: Sequence[dolfiny.inequality.Inequality] | Sequence[ufl.equation.Equation],
    Jg: Sequence[Sequence[ufl.Form | None]] | None,
    bcs: Sequence[dolfinx.fem.DirichletBC] = [],
    form_compiler_options: dict[str, str] | None = None,
    jit_options: dict[str, str] | None = None,
) -> tuple[  # type: ignore
    tuple[TAOConstraintsFunction, PETSc.Vec], tuple[TAOConstraintsJacobianFunction, PETSc.Mat]
]:
    """Create constraint and jacobian callback from form.

    Parameters
    ----------
    comm:
        Communicator that all functions have in common.
    x0:
        Vector associated with the optimisation - usually created by `wrap_objective_callbacks`.
    u:
        The list of arguments the form is defined over.
    g:
        Constraint expression, either equality or inequality.
    Jg:
        Jacobian of the constraint's non constant part (lhs).
    bcs:
        List of boundary conditions that the callbacks will respect.
    form_compiler_options:
        Option for form compilation, i.e. FFCx options.
    jit_options:
        Compiler flags to use during form compilation.

    Returns
    -------
    Tuple of wrapped constraint callback together with associated vector, and tuple of Jacobian
    callbkack together with the associated matrix.

    """
    if Jg is None:
        # Once multiple constraints supported, switch to
        # Jg = [[ufl.derivative(_g.lhs, _u) for _u in u] for _g in g]
        Jg = [[ufl.derivative(g[0].lhs, _u) for _u in u]]
        Jg = [[None if (e is None or e.empty()) else e for e in row] for row in Jg]

    def compile(form):
        return dolfinx.fem.form(
            form, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

    # Once multiple constraints supported, switch to
    # g = [(compile(_g.lhs), _g.rhs) for _g in g]
    g_forms = [(compile(g[0].lhs), g[0].rhs)]
    Jg_forms: list[list[dolfinx.fem.Form]] = compile(Jg)

    if len(g_forms) != 1 or len(Jg_forms) != 1:
        raise NotImplementedError("Packing of multiple constraints not supported.")

    (g_lhs, g_rhs) = g_forms[0]

    arity = g_lhs.rank
    if arity == 0:
        if len(g_forms) != 1:
            raise NotImplementedError("Blocked constraint system not supported.")

        constraint_count = len(g)
        g_vec = PETSc.Vec().createMPI([constraint_count if comm.rank == 0 else 0, 1], comm=comm)  # type: ignore
        g_vec.setUp()

        Jg_mat = PETSc.Mat().create(comm=comm)  # type: ignore
        constraint_count = len(g_forms)
        Jg_mat.setSizes(
            [[x0.getLocalSize(), x0.getSize()], [PETSc.DECIDE, constraint_count]]  # type: ignore
        )  # [[nrl, nrg], [ncl, ncg]]
        Jg_mat.setType("dense")
        Jg_mat.setUp()

        # Note: does not copy Jg_mat, the actual transpose is never built.
        Jg_mat_T = PETSc.Mat()  # type: ignore
        Jg_mat_T.createTranspose(Jg_mat)
        Jg_mat = Jg_mat_T  # TODO: document
    elif arity == 1:
        # TODO: fix blocking mess
        g_vec = dolfinx.fem.petsc.create_vector([g_lhs], kind="mpi")
        Jg_mat = dolfinx.fem.petsc.create_matrix(Jg_forms)
    else:
        raise TypeError(f"Constraint-lhs has arity {arity}. Only arity 0 and 1 are supported.")

    @sync_functions(u)
    def _g_callback(tao, x, c):
        (g_lhs, g_rhs) = g_forms[0]

        match g_lhs.rank:
            case 0:
                g_lhs_value = dolfinx.fem.assemble_scalar(g_lhs)
                g_lhs_value = comm.allreduce(g_lhs_value, MPI.SUM)

                if comm.rank == 0:
                    c[0] = g_lhs_value - g_rhs
            case 1:
                x.setAttr("_blocks", x0.getAttr("_blocks"))

                with c.localForm() as c_local:
                    c_local.set(0.0)

                dolfinx.fem.petsc.assemble_vector(c, g_lhs)
                dolfinx.fem.petsc.apply_lifting(
                    c,
                    Jg_forms,
                    bcs=dolfinx.fem.bcs_by_block(
                        dolfinx.fem.extract_function_spaces(Jg_forms, 1), bcs
                    ),
                    x0=x,
                    alpha=-1.0,
                )
                c.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                dolfinx.fem.petsc.set_bc(
                    c,
                    bcs=dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces([g_lhs]), bcs),
                    x0=x,
                    alpha=-1.0,
                )

                c.shift(-g_rhs)
            case _:
                raise RuntimeError()
        c.assemble()

    @sync_functions(u)
    def _Jg_callback(tao, x, J, P) -> None:
        _Jg: dolfinx.fem.Form = Jg_forms[0][0]

        match _Jg.rank:
            case 0:
                raise RuntimeError("Variation of constraint can not have arity 0.")
            case 1:
                JT = J.getTransposeMat()
                JT.zeroEntries()

                Jg_vec = dolfinx.fem.petsc.assemble_vector(_Jg)
                Jg_vec.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore

                offset = Jg_vec.getOwnershipRange()[0]
                local_size = Jg_vec.getLocalSize()
                JT.setValues(
                    np.arange(offset, offset + local_size, dtype=np.int32), [0], Jg_vec.getArray()
                )

                JT.assemble()
                JT.createTranspose(J)
            case 2:
                J.zeroEntries()
                dolfinx.fem.petsc.assemble_matrix(J, Jg_forms, bcs=bcs, diag=1.0)  # type: ignore
            case _:
                raise RuntimeError()

        J.assemble()

    return (_g_callback, g_vec), (_Jg_callback, Jg_mat)


class TAOProblem:
    _converged_reasons_tao = attributes_to_dict(PETSc.TAO.ConvergedReason, invert=True)  # type: ignore
    _converged_reasons_ksp = attributes_to_dict(PETSc.KSP.ConvergedReason, invert=True)  # type: ignore

    _x0: PETSc.Vec  # type: ignore
    _J: tuple[TAOJacobianFunction, PETSc.Vec] | None  # type: ignore
    _H: tuple[TAOHessianFunction, PETSc.Mat] | None  # type: ignore
    _g: tuple[TAOConstraintsFunction, PETSc.Vec] | None  # type: ignore
    _Jg: tuple[TAOConstraintsJacobianFunction, PETSc.Mat] | None  # type: ignore
    _h: tuple[TAOConstraintsFunction, PETSc.Vec] | None  # type: ignore
    _Jh: tuple[TAOConstraintsJacobianFunction, PETSc.Mat] | None  # type: ignore

    def __init__(
        self,
        F: TAOObjectiveFunction | ufl.Form,
        u: Sequence[dolfinx.fem.Function],
        bcs: Sequence[dolfinx.fem.DirichletBC] = [],
        lb: np.floating | Sequence[dolfinx.fem.Function] = PETSc.NINFINITY,  # type: ignore
        ub: np.floating | Sequence[dolfinx.fem.Function] = PETSc.INFINITY,  # type: ignore
        J: Sequence[ufl.Form] | tuple[TAOJacobianFunction, PETSc.Vec] | None = None,  # type: ignore
        H: Sequence[Sequence[ufl.Form]] | tuple[TAOHessianFunction, PETSc.Mat] | None = None,  # type: ignore
        g: Sequence[dolfiny.inequality.Inequality]  # type: ignore
        | tuple[TAOConstraintsFunction, PETSc.Vec]
        | None = None,
        Jg: Sequence[Sequence[ufl.Form]] | None = None,
        h: Sequence[ufl.equation.Equation] | tuple[TAOConstraintsFunction, PETSc.Vec] | None = None,  # type: ignore
        Jh: Sequence[Sequence[ufl.Form]] | None = None,
        prefix=None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ):
        """Create `TAOProblem`, which is a wrapper of `PETSc.TAO`.

        Only sets the problem up, does not solve it, see `TAOProblem.solve`.

        Parameters
        ----------
        F:
            Objective function.
        u:
            Arguments of the optimisation problem, also act as initial guess.
        bcs:
            Boundary conditions of the arguments.
        lb:
            Lower bound constraints.
        ub:
            Upper bound constraints.
        J:
            Jacobian of objective functional.
        H:
            Hessian of objective functional.
        g:
            Inequality constraints, see `dolfiny.inequality.Inequality` - constant rhs.
        Jg:
            Jacobian of inequality constraints.
        h:
            Equality constraints, see `ufl.equation.Equation` - constant rhs.
        Jh:
            Jacobian of equality constraints.
        prefix:
            Prefix for the PETSc options database.
        form_compiler_options:
            Option for form compilation, i.e. FFCx options.
        jit_options:
            Compiler flags to use during form compilation.

        """
        if len(u) == 0:
            raise RuntimeError("List of provided variable sequence is empty.")

        if not all(isinstance(_u, dolfinx.fem.Function) for _u in u):
            raise RuntimeError("Provided variables not of type dolfinx.fem.Function.")

        self._F = F
        self._u = u
        self._bcs = bcs

        self._comm = self._u[0].function_space.mesh.comm
        # TODO: check for eq/congruent comms

        if isinstance(self._F, ufl.Form):
            # TODO: do not override J and H

            self._x0, self._F, self._J, self._H = wrap_objective_callbacks(
                self._comm,
                self._u,
                self._F,
                J,  # type: ignore
                H,  # type: ignore
                self._bcs,
                form_compiler_options,
                jit_options,
            )
        else:
            # direct objective
            assert len(self._u) == 1  # TODO!!
            self._x0 = dolfinx.la.petsc.create_vector(
                self._u[0].function_space.dofmap.index_map,
                self._u[0].function_space.dofmap.index_map_bs,
            )
            self._J = J  # type: ignore
            self._H = H  # type: ignore

        self._tao = PETSc.TAO().create(self._comm)  # type: ignore
        self._tao.setOptionsPrefix(prefix)
        self._tao.setFromOptions()
        self._tao.setSolution(self._x0)
        self._tao.setObjective(self._F)

        def _create_bounds_vec(b):
            b_vec = self._x0.copy()
            if isinstance(b, float | int):
                b_vec.set(b)
            else:
                dolfinx.fem.petsc.assign(b, b_vec)
                b_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            return b_vec

        self._tao.setVariableBounds(tuple(_create_bounds_vec(b) for b in (lb, ub)))

        if self._J is not None:
            self._tao.setGradient(*self._J)

        if self._H is not None:
            self._tao.setHessian(*self._H)

        if g is not None:
            # TODO: check _g either callback, or sequence of equations
            assert len(g) > 0
            if isinstance(g[0], ufl.equation.Equation):
                assert len(g) == 1  # TODO
                self._g, self._Jg = wrap_constraint_callbacks(
                    self._comm,
                    self._x0,
                    self._u,
                    g,
                    Jg,
                    self._bcs,
                    form_compiler_options,
                    jit_options,
                )
            else:
                self._g = g  # type: ignore
                self._Jg = Jg  # type: ignore

            self._tao.setEqualityConstraints(*self._g)
            self._tao.setJacobianEquality(*self._Jg)
        else:
            self._g = None
            self._Jg = None

        if h is not None:
            # TODO: check _h either callback, or sequence of inequalities
            if isinstance(h[0], dolfiny.inequality.Inequality):
                self._h, self._Jh = wrap_constraint_callbacks(
                    self._comm,
                    self._x0,
                    self._u,
                    h,
                    Jh,
                    self._bcs,
                    form_compiler_options,
                    jit_options,
                )
            else:
                self._h = h  # type: ignore
                self._Jh = Jh  # type: ignore

            self._tao.setInequalityConstraints(*self._h)
            self._tao.setJacobianInequality(*self._Jh)
        else:
            self._h = None
            self._Jh = None

        self._tao.setUp()
        self._tao.setMonitor(self._monitor)

        if self._tao.getType() == PETSc.TAO.Type.ALMM:  # type: ignore
            if (PETSc.Sys().getVersion()[1] >= 23) and (PETSc.Sys().getVersion()[2] >= 3):  # type: ignore
                subsolver = self._tao.getALMMSubsolver()
                subsolver.setMonitor(self._monitor)

        if ksp := self._tao.getKSP():
            ksp.setMonitor(self._monitor_ksp)

    def solve(self) -> None:
        """Solve the optimisation problem."""
        dolfinx.fem.petsc.assign(self._u, self._x0)
        self._x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

        self._tao.solve(self._x0)

        solution = self._tao.getSolution()
        # TODO: code duplication with link_state -> resolve
        solution.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        dolfinx.fem.petsc.assign(solution, self._u)

    @property
    def tao(self) -> PETSc.TAO:  # type: ignore
        """Return the underlying `PETSc.TAO` object."""
        return self._tao

    @property
    def u(self) -> Sequence[dolfinx.fem.Function]:
        """Return the arguments/solutions of the optimisation problem."""
        return self._u

    def destroy(self) -> None:
        self._tao.destroy()
        self._x0.destroy()
        if self._J is not None:
            self._J[1].destroy()
        if self._H is not None:
            self._H[1].destroy()
        if self._h is not None:
            self._h[1].destroy()
        if self._Jh is not None:
            self._Jh[1].destroy()
        if self._g is not None:
            self._g[1].destroy()
        if self._Jg is not None:
            self._Jg[1].destroy()

    def __del__(self) -> None:
        self.destroy()

    def _monitor_ksp(self, ksp, ksp_it, ksp_norm):
        it = self.tao.getIterationNumber()
        message = "\033[90m"  # bright black
        ksp_info = TAOProblem._converged_reasons_ksp[ksp.reason]
        message += f"# TAO {it:3d}, KSP {ksp_it:3d}      |r|={ksp_norm:9.3e} ({ksp_info:s})"
        message += ANSI.reset
        pprint(message)

    def _monitor(self, tao: PETSc.TAO):  # type: ignore
        color = ANSI.blue if tao.getType() == PETSc.TAO.Type.ALMM else ""  # type: ignore
        it = tao.getIterationNumber()
        reason_s = TAOProblem._converged_reasons_tao[tao.reason]
        status_color = ANSI.red if tao.reason > 0 else ""
        pprint(f"{color}# TAO {it:3d} ({status_color}{reason_s}{color}){ANSI.reset}")

        # TODO: ls (limited by https://gitlab.com/petsc/petsc/-/merge_requests/8456)
        # TODO: snes (once PDIPM available)

        blocks = self._x0.getAttr("_blocks")
        for i, u in enumerate(self._u):
            # TODO: this breaks
            # message = f"# sub {i:1d} |x|={u.x.petsc_vec.norm():9.3e}"
            s = u.x.index_map.size_local * u.x.block_size
            message = f"{color}# sub   {i:1d} [{prefixify(s):s}] |x|={dolfinx.la.norm(u.x):9.3e}"
            if self._J is not None:
                if blocks is not None:
                    offset0 = blocks[0][i]
                    offset1 = blocks[1][i]
                    with self._J[1].localForm() as J_local:
                        arr = J_local.array[offset0:offset1]
                        norm = np.sqrt(self._comm.allreduce(np.inner(arr, arr)))
                    message += f" |J|={norm:9.3e}"
                else:
                    message += f" |J|={self._J[1].norm():9.3e}"

            message += f" ({u.name}){ANSI.reset}"
            pprint(message)

        message = f"{color}# all            |x|={self._x0.norm():9.3e}"
        if self._J is not None:
            message += f" |J|={self._J[1].norm():9.3e}"

        if self._H is not None and self._H[1].isAssembled():
            message += f" |H|={self._H[1].norm():9.3e}"

        if self._h is not None:
            message += f" |h|={self._h[1].norm():9.3e}"

        if self._Jh is not None:
            if self._Jh[1].getType() == PETSc.Mat.Type.TRANSPOSE:  # type: ignore
                message += f" |Jh|={self._Jh[1].getTransposeMat().norm():9.3e}"
            else:
                message += f" |Jh|={self._Jh[1].norm():9.3e}"

        if self._g is not None:
            message += f" |g|={self._g[1].norm():9.3e}"

        if self._Jg is not None:
            if self._Jg[1].getType() == PETSc.Mat.Type.TRANSPOSE:  # type: ignore
                message += f" |Jg|={self._Jg[1].getTransposeMat().norm():9.3e}"
            else:
                message += f" |Jg|={self._Jg[1].norm():9.3e}"

        message += f" f={tao.getFunctionValue():9.3e}"
        message += ANSI.reset
        pprint(message)

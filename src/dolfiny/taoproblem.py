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

import numpy as np

import dolfiny
import dolfiny.inequality


def sync_functions(u: Sequence[dolfinx.fem.Function]):
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
    H: Sequence[Sequence[ufl.Form]] | None = None,
    bcs: Sequence[dolfinx.fem.DirichletBC] = [],
    form_compiler_options: dict[str, str] | None = None,
    jit_options: dict[str, str] | None = None,
) -> tuple[  # type: ignore
    PETSc.Vec,
    TAOObjectiveFunction,
    tuple[TAOJacobianFunction, PETSc.Vec],
    tuple[TAOHessianFunction, PETSc.Mat],
]:
    if J is None:
        δu = ufl.TestFunctions(ufl.MixedFunctionSpace(*(_u.function_space for _u in u)))
        J = ufl.derivative(F, u, δu)
        J = ufl.extract_blocks(J)

    if H is None:
        assert J is not None

        H = [[ufl.derivative(_J, _u, ufl.TrialFunction(_u.function_space)) for _u in u] for _J in J]
        H = [[None if e.empty() else e for e in row] for row in H]

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
        F_value: float = dolfinx.fem.assemble_scalar(F_form)
        F_value = comm.allreduce(F_value, op=MPI.SUM)
        return F_value

    @sync_functions(u)
    def _callback_J(tao: PETSc.TAO, x: PETSc.Vec, J_vec: PETSc.Vec) -> None:  # type: ignore
        x.setAttr("_blocks", x0.getAttr("_blocks"))
        J_vec.setAttr("_blocks", x0.getAttr("_blocks"))
        J_vec.zeroEntries()
        J_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

        dolfinx.fem.petsc.assemble_vector(J_vec, J_form)

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
    g: ufl.Form,
    Jg: Sequence[ufl.Form] | None,
    bcs: Sequence[dolfinx.fem.DirichletBC] = [],
    form_compiler_options: dict[str, str] | None = None,
    jit_options: dict[str, str] | None = None,
) -> tuple[  # type: ignore
    tuple[TAOConstraintsFunction, PETSc.Vec], tuple[TAOConstraintsJacobianFunction, PETSc.Mat]
]:
    if Jg is None:
        Jg = [[ufl.derivative(_g.lhs, _u) for _u in u] for _g in g]
        Jg = [[None if e.empty() else e for e in row] for row in Jg]

    def compile(form):
        return dolfinx.fem.form(
            form, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

    g = [(compile(_g.lhs), _g.rhs) for _g in g]
    Jg = [compile(_Jg) for _Jg in Jg]

    if len(g) != 1 or len(Jg) != 1:
        raise NotImplementedError("Packing of multiple constraints not supported.")

    (g_lhs, g_rhs) = g[0]

    arity = g_lhs.rank
    if arity == 0:
        if len(g) != 1:
            raise NotImplementedError("Blocked constraint system not supported.")

        constraint_count = len(g)
        g_vec = PETSc.Vec().createMPI(  # type: ignore
            [constraint_count if comm.rank == 0 else 0, 1], comm=comm
        )
        g_vec.setUp()

        Jg_mat = PETSc.Mat().create(comm=comm)  # type: ignore
        constraint_count = len(g)
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
        Jg_mat = dolfinx.fem.petsc.create_matrix(Jg)
    else:
        raise TypeError(f"Constraint-lhs has arity {arity}. Only arity 0 and 1 are supported.")

    @sync_functions(u)
    def _g_callback(tao, x, c):
        (g_lhs, g_rhs) = g[0]

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
                    Jg,
                    bcs=dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(Jg, 1), bcs),
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
        _Jg = Jg[0][0]

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
                dolfinx.fem.petsc.assemble_matrix(J, Jg, bcs=bcs, diag=1.0)  # type: ignore
            case _:
                raise RuntimeError()

        J.assemble()

    return (_g_callback, g_vec), (_Jg_callback, Jg_mat)


class TAOProblem:
    def __init__(
        self,
        F: TAOObjectiveFunction | ufl.Form,
        u: Sequence[dolfinx.fem.Function],
        bcs: Sequence[dolfinx.fem.DirichletBC] = [],
        lb: float | Sequence[dolfinx.fem.Function] = PETSc.NINFINITY,  # type: ignore
        ub: float | Sequence[dolfinx.fem.Function] = PETSc.INFINITY,  # type: ignore
        J: Sequence[dolfinx.fem.Form] | None = None,
        H: Sequence[Sequence[dolfinx.fem.Form]] | None = None,
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
        if len(u) == 0:
            raise RuntimeError("List of provided variable sequence is empty.")

        if not all(isinstance(_u, dolfinx.fem.Function) for _u in u):
            raise RuntimeError("Provided variables not of type dolfinx.fem.Function.")

        self._F = F
        self._u = u
        self._bcs = bcs
        self._J = J
        self._H = H

        self._comm = self._u[0].function_space.mesh.comm
        # TODO: check for eq/congruent comms

        if isinstance(self._F, ufl.Form):
            # TODO: do not override J and H

            self._x0, self._F, self._J, self._H = wrap_objective_callbacks(  # type: ignore
                self._comm,
                self._u,
                self._F,
                self._J,
                self._H,
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

    @property
    def tao(self) -> PETSc.TAO:  # type: ignore
        return self._tao

    @property
    def u(self) -> Sequence[dolfinx.fem.Function]:
        return self._u

    def solve(self):
        # TODO: monitor

        dolfinx.fem.petsc.assign(self._u, self._x0)
        self._x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self._tao.solve(self._x0)

        solution = self._tao.getSolution()
        # TODO: code duplication with link_state -> resolve
        solution.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        dolfinx.fem.petsc.assign(solution, self._u)

        return self._u

from collections.abc import Callable, Sequence

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.typing import (
    TAOConstraintsFunction,
    TAOHessianFunction,
    TAOJacobianFunction,
    TAOObjectiveFunction,
)

import dolfinx
import dolfinx.fem.petsc
import dolfinx.la.petsc
import ufl

import dolfiny
import dolfiny.inequality
from dolfiny.function import functions_to_vec, vec_to_functions

# replace after: https://gitlab.com/petsc/petsc/-/merge_requests/8342 is merged
TAOConstraintsJacobianFunction = Callable[[PETSc.TAO, PETSc.Vec, PETSc.Mat, PETSc.Mat], None]  # type: ignore


# TODO: improve naming, especially for PDE constrained state implies something incorrect here
# TOOD: can we make x disapper form callback? No miss interpretation possible?
def link_state(u: Sequence[dolfinx.fem.Function]):
    # T = TypeVar("T", bound=Callable[[], float | None])
    def _decorator[T](_to_wrap: T) -> T:
        def _wrapped_callback(tao: PETSc.TAO, x: PETSc.Vec, *args) -> float | None:  # type: ignore
            # Update the underlying states in u
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
            dolfinx.fem.petsc.assign(x, u)

            return _to_wrap(tao, x, *args)  # type: ignore

        return _wrapped_callback  # type: ignore

    return _decorator


class TAOBlockProblem:
    @staticmethod
    def wrap_form_objective(
        comm: MPI.Comm,
        u: Sequence[dolfinx.fem.Function],
        F: ufl.Form,
        J: Sequence[ufl.Form] | None = None,
        H: Sequence[Sequence[ufl.Form]] | None = None,
        bcs: Sequence[dolfinx.fem.DirichletBC] | None = None,
        form_compiler_options: dict[str, str] | None = None,
        jit_options: dict[str, str] | None = None,
    ) -> tuple[
        PETSc.Vec,
        TAOObjectiveFunction,
        tuple[TAOJacobianFunction, PETSc.Vec],
        tuple[TAOHessianFunction, PETSc.Mat],
    ]:
        if J is None:
            δu = [ufl.TestFunction(_u.function_space) for _u in u]
            J = dolfiny.expression.derivative(F, u, δu)
            J = dolfiny.function.extract_blocks(J, δu)

        if H is None:
            # TODO:
            # δu = [ufl.TestFunction(_u.function_space) for _u in self._u]
            # δδu = [ufl.TrialFunction(_u.function_space) for _u in self._u]

            # self._H = dolfiny.expression.derivative(self._F, self._u, δu)
            # self._H = dolfiny.expression.derivative(self._H, self._u, δδu)
            # self._H = dolfiny.function.extract_blocks(self._H, δu, δδu)

            H = [
                [ufl.derivative(_J, _u, ufl.TrialFunction(_u.function_space)) for _u in u]
                for _J in J
            ]
            H = [[None if e.empty() else e for e in row] for row in H]

        def compile(form):
            return dolfinx.fem.form(
                form, form_compiler_options=form_compiler_options, jit_options=jit_options
            )

        F = compile(F)
        J = compile(J)
        H = compile(H)

        x0 = dolfinx.fem.petsc.create_vector(J, kind=PETSc.Vec.Type.MPI)

        jacobian = x0.copy()
        jacobian.setAttr("_blocks", x0.getAttr("_blocks"))

        hessian = dolfinx.fem.petsc.create_matrix(H)

        @link_state(u)
        def _callback_F(tao: PETSc.TAO, x: PETSc.Vec) -> float:  # type: ignore
            local_F = dolfinx.fem.assemble_scalar(F)
            F_value = comm.allreduce(local_F, op=MPI.SUM)
            return F_value

        @link_state(u)
        def _callback_J(tao: PETSc.TAO, x: PETSc.Vec, J_vec: PETSc.Vec) -> None:  # type: ignore
            x.setAttr("_blocks", x0.getAttr("_blocks"))

            J_vec.zeroEntries()
            J_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

            dolfinx.fem.petsc.assemble_vector(J_vec, J)

            if bcs is not None:
                dolfinx.fem.petsc.apply_lifting(
                    J_vec,
                    H,
                    bcs=dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(H, 1), bcs),
                    x0=x,
                    alpha=-1.0,
                )

            J_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore

            if bcs is not None:
                dolfinx.fem.petsc.set_bc(
                    J_vec,
                    bcs=dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(J), bcs),
                    x0=x,
                    alpha=-1.0,
                )

        @link_state(u)
        def _callback_H(tao: PETSc.TAO, x: PETSc.Vec, H_mat: PETSc.Mat, P_mat: PETSc.Mat) -> None:  # type: ignore
            x.setAttr("_blocks", x0.getAttr("_blocks"))
            H_mat.zeroEntries()

            dolfinx.fem.petsc.assemble_matrix(H_mat, H, bcs, diag=1.0)

            H_mat.assemble()

            if P_mat != H:
                H_mat.copy(P_mat)
                P_mat.assemble()

        return x0, _callback_F, (_callback_J, jacobian), (_callback_H, hessian)

    @staticmethod
    def wrap_form_constraints(
        comm: MPI.Comm,
        x0: PETSc.Vec,  # type: ignore
        u: Sequence[dolfinx.fem.Function],
        g: ufl.Form,
        Jg: Sequence[ufl.Form] | None,
        bcs: Sequence[dolfinx.fem.DirichletBC] | None = None,
        form_compiler_options: dict[str, str] | None = None,
        jit_options: dict[str, str] | None = None,
    ) -> tuple[
        tuple[TAOConstraintsFunction, PETSc.Vec], tuple[TAOConstraintsJacobianFunction, PETSc.Vec]
    ]:
        if Jg is None:
            raise NotImplementedError("Can not auto diff constraints yet")

        def compile(form):
            return dolfinx.fem.form(
                form, form_compiler_options=form_compiler_options, jit_options=jit_options
            )

        g = [(compile(_g.lhs), _g.rhs) for _g in g]
        Jg = [compile(_Jg) for _Jg in Jg]

        assert len(g) == 1  # TODO!

        (g_lhs, g_rhs) = g[0]
        # arities = ufl.algorithms.formtransformations.compute_form_arities(g_lhs)
        # if len(arities) != 1:
        #     raise TypeError("Equality constraint form has multiple (or none) arities.")

        # arity = next(iter(arities))

        arity = g_lhs.rank

        match arity:
            case 0:
                constraint_count = len(g)
                g_vec = PETSc.Vec().createMPI(  # type: ignore
                    [constraint_count if MPI.COMM_WORLD.rank == 0 else 0, 1], comm=MPI.COMM_WORLD
                )
                g_vec.setUp()

                # TODO: shorten/remove?
                g_vec.zeroEntries()
                g_vec.assemble()
            case 1:
                g_vec = dolfinx.fem.petsc.create_vector([g_lhs], kind="mpi")
            case _:
                raise TypeError(
                    f"Constraint-lhs has arity {arity}. Only arity 0 and 1 are supported."
                )

        match g_lhs.rank:
            case 0:
                Jg_mat = PETSc.Mat().create(comm=MPI.COMM_WORLD)  # type: ignore
                # TODO: should all k rows belong to fist process? implications?
                constraint_count = len(g)
                Jg_mat.setSizes(
                    [[x0.getLocalSize(), x0.getSize()], [PETSc.DECIDE, constraint_count]]  # type: ignore
                )  # [[nrl, nrg], [ncl, ncg]]
                Jg_mat.setType("dense")
                Jg_mat.setUp()
                Jg_mat_T = PETSc.Mat()  # type: ignore
                Jg_mat_T.createTranspose(Jg_mat)
                Jg_mat = Jg_mat_T  # TODO: document
            case 1:
                Jg_mat = dolfinx.fem.petsc.create_matrix(Jg)

        @link_state(u)
        def _g_callback(tao, x, c):
            assert len(g) == 1  # TODO!

            i = 0
            (g_lhs, g_rhs) = g[i]

            # for i, (h, C) in enumerate(self._g):
            match g_lhs.rank:
                case 0:
                    h_value = dolfinx.fem.assemble_scalar(g_lhs)
                    h_value = MPI.COMM_WORLD.allreduce(h_value, MPI.SUM)

                    if MPI.COMM_WORLD.rank == 0:
                        # TODO: not just i?
                        c[i] = h_value - g_rhs
                case 1:
                    x.setAttr("_blocks", x0.getAttr("_blocks"))

                    with c.localForm() as c_local:
                        c_local.set(0.0)

                    dolfinx.fem.petsc.assemble_vector(c, g_lhs)
                    dolfinx.fem.petsc.apply_lifting(
                        c,
                        Jg,
                        bcs=dolfinx.fem.bcs_by_block(
                            dolfinx.fem.extract_function_spaces(Jg, 1), bcs
                        ),
                        x0=x,
                        alpha=-1.0,
                    )
                    c.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                    dolfinx.fem.petsc.set_bc(
                        c,
                        bcs=dolfinx.fem.bcs_by_block(
                            dolfinx.fem.extract_function_spaces([g_lhs]), bcs
                        ),
                        x0=x,
                        alpha=-1.0,
                    )
                    # c.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    # # type: ignore

                    c.shift(-g_rhs)
                case _:
                    raise RuntimeError()
            c.assemble()

            # u[0].x.array[:] = c.getArray()
            # with dolfinx.io.VTXWriter(u[0].function_space.mesh.comm, "Jg.bp", u[0], "bp4") as
            # file:
            #     file.write(0.0)
            # exit()

        @link_state(u)
        def _Jg_callback(tao, x, J, P) -> None:
            assert len(Jg) == 1  # TODO!

            if Jg[0][0].rank == 2:
                J.zeroEntries()
                # JT = J.getTransposeMat()
                # JT.zeroEntries()
                dolfinx.fem.petsc.assemble_matrix(J, Jg, bcs=bcs, diag=1.0)
                # JT.assemble()
                # JT.createTranspose(J)
                # J.assemble()
            else:
                JT = J.getTransposeMat()
                JT.zeroEntries()
                # not necessary since matrix
                # J.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

                for k, _Jg in enumerate(Jg[0]):
                    match _Jg.rank:
                        case 0:
                            raise RuntimeError("Variation of constraint can not have arity 0.")
                        case 1:
                            Jg_vec = dolfinx.fem.petsc.assemble_vector(_Jg)
                            Jg_vec.ghostUpdate(
                                addv=PETSc.InsertMode.ADD,
                                mode=PETSc.ScatterMode.REVERSE,  # type: ignore
                            )

                            offset = Jg_vec.getOwnershipRange()[0]
                            for i in range(Jg_vec.getLocalSize()):
                                JT.setValue(offset + i, k, Jg_vec.getArray()[i])
                            JT.assemble()
                        case 2:
                            pass
                            # TODO: move here
                            # dolfinx.fem.petsc.assemble_matrix(JT, _Jg)
                            # JT.assemble()
                        case _:
                            raise RuntimeError()

                # J.copy(JT)
                JT.createTranspose(J)
            J.assemble()

        return (_g_callback, g_vec), (_Jg_callback, Jg_mat)

    def __init__(
        self,
        F: TAOObjectiveFunction | ufl.Form,
        u: Sequence[dolfinx.fem.Function],
        bcs: Sequence[dolfinx.fem.DirichletBC] = [],
        J: Sequence[dolfinx.fem.Form] | None = None,
        H: Sequence[Sequence[dolfinx.fem.Form]] | None = None,
        g: Sequence[ufl.Form] | None = None,
        Jg: Sequence[ufl.Form] | None = None,
        h: Sequence[tuple[dolfinx.fem.Form, float]] | None = None,
        Jh: Sequence[dolfinx.fem.Form] | None = None,
        prefix=None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        # nest=False,
    ):
        """TODO: proper docstring...

        h = [(h_1, C_1), ..., (h_k, C_k)], encoding float constraints of shape

            h_i(x) >= C_i   for each i

        """
        if len(u) == 0:
            raise RuntimeError("List of provided variable sequence is empty.")

        if not all(isinstance(_u, dolfinx.fem.Function) for _u in u):
            raise RuntimeError("Provided variables not of type dolfinx.fem.Function.")

        self._F = F
        self._u = u
        self._bcs = bcs
        self._J = J
        self._H = H
        self._g = g
        self._Jg = Jg
        self._h = h
        self._Jh = Jh

        self._comm = self._u[0].function_space.mesh.comm
        # TODO: check for eq/congruent comms

        if isinstance(self._F, ufl.Form):
            # TODO: assert not
            # if not isinstance(self._J)

            self._x0, self._F, self._J, self._H = TAOBlockProblem.wrap_form_objective(
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
            # self._x0 = self._u[0].x.petsc_vec.copy()
            # self._x0.setAttr("_blocks", self._u[0].x.petsc_vec.getAttr("_blocks"))
            # print(self._u[0].x.petsc_vec.getAttr("_blocks"))
            # exit()
            # gradient = self._x0.copy()  # TODO: move

        self._tao = PETSc.TAO().create(self._comm)  # type: ignore
        self._tao.setOptionsPrefix(prefix)
        self._tao.setFromOptions()
        self._tao.setObjective(self._F)

        # TODO: input - also as scalar only
        lb = self._x0.copy()
        lb.set(-100)
        lb.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

        ub = self._x0.copy()
        ub.set(100)
        ub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

        self._tao.setVariableBounds((lb, ub))

        if self._J is not None:
            self._tao.setGradient(*self._J)

        if self._H is not None:
            self._tao.setHessian(*self._H)

        if self._g is not None:
            assert len(self._g) > 0
            if isinstance(self._g[0], ufl.equation.Equation):
                assert len(self._g) == 1  # TODO
                self._g, self._Jg = TAOBlockProblem.wrap_form_constraints(
                    self._comm,
                    self._x0,
                    self._u,
                    self._g,
                    self._Jg,
                    self._bcs,
                    form_compiler_options,
                    jit_options,
                )

            self._tao.setEqualityConstraints(*self._g)  # TODO
            self._tao.setJacobianEquality(*self._Jg)  # TODO

        if self._h is not None:
            if isinstance(self._h[0], dolfiny.inequality.Inequality):
                self._h, self._Jh = TAOBlockProblem.wrap_form_constraints(
                    self._comm,
                    self._x0,
                    self._u,
                    self._h,
                    self._Jh,
                    self._bcs,
                    form_compiler_options,
                    jit_options,
                )

            self._tao.setInequalityConstraints(*self._h)
            self._tao.setJacobianInequality(*self._Jh)

    @property
    def tao(self) -> PETSc.TAO:  # type: ignore
        return self._tao

    @property
    def u(self) -> Sequence[dolfinx.fem.Function]:
        return self._u

    def solve(self, u_init=None):
        if u_init is not None:
            functions_to_vec(u_init, self._x0)

        # TODO: monitor

        self._tao.solve(self._x0)
        # ls = self._tao.getLineSearch()
        self._tao.view()
        # ls.view()

        solution = [dolfinx.fem.Function(u.function_space, name=u.name) for u in self._u]
        vec_to_functions(self._x0, solution)
        return solution

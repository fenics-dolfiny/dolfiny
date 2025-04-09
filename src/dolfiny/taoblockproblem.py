from collections.abc import Sequence

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc

from dolfiny.function import functions_to_vec, vec_to_functions


class TAOBlockProblem:
    def __init__(
        self,
        F: Sequence[dolfinx.fem.Form],
        u: list,
        bcs: Sequence[dolfinx.fem.DirichletBC] = [],
        J: Sequence[dolfinx.fem.Form] | None = None,
        H: Sequence[dolfinx.fem.Form] | None = None,
        prefix=None,
        # nest=False,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ):
        if len(F) == 0:
            raise RuntimeError("List of provided residual forms is empty!")

        if len(u) == 0:
            raise RuntimeError("List of provided solution functions is empty!")

        if not all(isinstance(_u, dolfinx.fem.Function) for _u in u):
            raise RuntimeError("Provided solution function not of type dolfinx.Function!")

        self._F = F
        self._J = J
        self._H = H
        self._u = u
        self._bcs = bcs

        # TODO: check for eq/congruent comms
        self._comm = self._u[0].function_space.mesh.comm

        if self._J is None:
            raise NotImplementedError()

        if self._H is None:
            raise NotImplementedError()

        self.F_form_all_ufc = dolfinx.fem.form(
            self._F, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

        self.J_form_all_ufc = dolfinx.fem.form(
            self._J, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

        self.H_form_all_ufc = dolfinx.fem.form(
            self._H, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

        # By default, copy all forms as the forms used in assemblers
        self._F = self.F_form_all_ufc.copy()
        self._J = self.J_form_all_ufc.copy()
        self._H = self.H_form_all_ufc.copy()

        # TOOD: remove
        if self._H is None:
            raise NotImplementedError()

        self._x0 = self._u[0].x.petsc_vec.copy()
        self._tao = PETSc.TAO().create(self._comm)  # type: ignore
        self._tao.setOptionsPrefix(prefix)
        self._tao.setFromOptions()
        self._tao.setObjective(self._F_callback)
        self._tao.setGradient(self._J_callback, self._u[0].x.petsc_vec.copy())
        self._tao.setHessian(self._H_callback, dolfinx.fem.petsc.create_matrix(self._H[0]))

    def _F_callback(self, tao, x):
        vec_to_functions(x, self._u)

        assert len(self._F) == 1  # TODO: extend for blocked problems

        local_F = dolfinx.fem.assemble_scalar(self._F[0])
        F_value = self._comm.allreduce(local_F, op=MPI.SUM)
        return F_value

    def _J_callback(self, tao, x, J):
        J.zeroEntries()
        J.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        vec_to_functions(x, self._u)

        dolfinx.fem.petsc.assemble_vector(J, self._J[0])
        J.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        dolfinx.fem.petsc.apply_lifting(
            J,
            self._H,
            bcs=dolfinx.fem.bcs_by_block(
                dolfinx.fem.extract_function_spaces([self._H], 1), self._bcs
            ),
            x0=[x],
            alpha=-1.0,
        )
        dolfinx.fem.petsc.set_bc(
            J,
            bcs=dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(self._J), self._bcs)[
                0
            ],
            x0=x,
            alpha=-1.0,
        )

    def _H_callback(self, tao, x, H, P):
        H.zeroEntries()
        vec_to_functions(x, self._u)

        dolfinx.fem.petsc.assemble_matrix(H, [self._H], self._bcs, diag=1.0)

        H.assemble()

        if P != H:
            H.copy(P)
            P.assemble()

    def solve(self, u_init=None):
        if u_init is not None:
            functions_to_vec(u_init, self._x0)

        # TODO: monitor

        self._tao.solve(self._x0)
        ls = self._tao.getLineSearch()
        self._tao.view()
        ls.view()

        solution = [dolfinx.fem.Function(u.function_space, name=u.name) for u in self._u]
        vec_to_functions(self._x0, solution)
        return solution

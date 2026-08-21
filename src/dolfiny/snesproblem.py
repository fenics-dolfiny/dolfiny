import logging
from collections.abc import Sequence
from typing import cast

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import ufl
from dolfinx.mesh import EntityMap

import numpy as np

from dolfiny.localsolver import LocalSolver
from dolfiny.utils import ANSI, attributes_to_dict, prefixify

logger = logging.getLogger(__name__)


class SNESProblem:
    reasons_ksp = attributes_to_dict(PETSc.KSP.ConvergedReason, invert=True)
    reasons_snes = attributes_to_dict(PETSc.SNES.ConvergedReason, invert=True)

    def __init__(
        self,
        F_form: list[ufl.Form],
        u: list[dolfinx.fem.Function],
        bcs: list[dolfinx.fem.DirichletBC] = [],
        J_form: list[list[ufl.Form | None]] | None = None,
        nest: bool = False,
        restriction=None,
        prefix: str | None = None,
        localsolver: LocalSolver | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: Sequence[EntityMap] | None = None,
    ):
        """SNES problem and solver wrapper.

        Parameters
        ----------
        F_form
            Residual forms
        u
            Current solution functions
        bcs
            List of boundary conditions
        J_form
            Jacobian forms
        nest: False
            True for 'matnest' data layout, False for 'aij'
        restriction: optional
            ``Restriction`` class used to provide information about degree-of-freedom
            indices for which this solver should solve. With ``LocalSolver`` only
            degrees-of-freedom for global fields must be considered.
        prefix
            Prefix for the PETSc options database
        localsolver: optional
            ``LocalSolver`` class providing context on elimination of local
            degrees-of-freedom.
        form_compiler_options
            Forwarded to FFCx
        jit_options
            Compiler flags to use for form compilation
        entity_maps
            Entity maps to pass to the form. Needs to be provided when submeshes are used.

        """
        self.u = u

        if not len(F_form) > 0:
            raise RuntimeError("List of provided residual forms is empty!")

        if not len(self.u) > 0:
            raise RuntimeError("List of provided solution functions is empty!")

        if not all(isinstance(_u, dolfinx.fem.Function) for _u in u):
            raise RuntimeError("Provided solution function not of type dolfinx.Function!")

        self.comm = self.u[0].function_space.mesh.comm

        if J_form is None:
            J_form = [[None for i in range(len(self.u))] for j in range(len(self.u))]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    uj = self.u[j]
                    duj = ufl.TrialFunction(uj.function_space)
                    J_form[i][j] = ufl.derivative(F_form[i], uj, duj)

                    # If the form happens to be empty replace with None
                    if J_form[i][j] is not None and J_form[i][j].empty():  # type: ignore[union-attr]
                        J_form[i][j] = None

        # Compile all forms
        self.F_form_all_ufc = dolfinx.fem.form(
            F_form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )
        self.J_form_all_ufc = dolfinx.fem.form(
            J_form,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        # By default, copy all forms as the forms used in assemblers
        assert not any(form is None for form in self.F_form_all_ufc)
        self.F_form = cast(list[dolfinx.fem.Form], self.F_form_all_ufc.copy())
        self.J_form = self.J_form_all_ufc.copy()
        self.global_spaces_id = range(len(self.u))

        self.nest = nest
        self.restriction = restriction
        self.localsolver = localsolver

        if self.localsolver is not None:
            # Set UFL and compiled forms to localsolver
            self.localsolver.F_ufl = F_form.copy()
            self.localsolver.J_ufl = J_form.copy()
            self.localsolver.F_ufc = self.F_form_all_ufc.copy()
            self.localsolver.J_ufc = self.J_form_all_ufc.copy()

            # Stack Coefficients and Constants
            self.localsolver.stack_data()

            # Replace compiled forms with wrapped forms for local solver
            self.F_form = self.localsolver.reduced_F_forms()
            self.J_form = self.localsolver.reduced_J_forms()
            self.local_form = self.localsolver.local_form()
            self.global_spaces_id = self.localsolver.global_spaces_id

        self.bcs = bcs

        # Prepare empty functions on the corresponding sub-spaces
        # These store solution sub-functions
        self.solution = [dolfinx.fem.Function(u.function_space, name=u.name) for u in self.u]

        self.norm_r: dict[int, np.ndarray] = {}
        self.norm_dx: dict[int, np.ndarray] = {}
        self.norm_x: dict[int, np.ndarray] = {}
        self.size_x: dict[int, np.ndarray] = {}

        self.snes = PETSc.SNES().create(self.comm)  # type: ignore[arg-type]

        if self.nest:
            if self.restriction is not None:
                raise RuntimeError("Restriction for MATNEST not yet supported.")

            if self.localsolver is not None:
                raise RuntimeError("LocalSolver for MATNEST not yet supported.")

            self.J = dolfinx.fem.petsc.create_matrix(self.J_form, kind=PETSc.Mat.Type.NEST)
            self.F = dolfinx.fem.petsc.create_vector(
                dolfinx.fem.extract_function_spaces(self.F_form), kind=PETSc.Vec.Type.NEST
            )
            self.x = self.F.copy()
            self.x0 = self.F.copy()

            self.snes.setFunction(self._F_nest, self.F)
            self.snes.setJacobian(self._J_nest, self.J)

        else:
            if self.localsolver is not None:
                # Create global vector where all local fields are assembled into
                # TODO: this might be a bug in dolfinx: doc claims None and MPI should have same
                #       behavior, None does not work atm.
                self.xloc = dolfinx.fem.petsc.create_vector(
                    dolfinx.fem.extract_function_spaces(self.local_form),
                    kind=PETSc.Vec.Type.MPI,
                )

            self.J = dolfinx.fem.petsc.create_matrix(self.J_form)
            # TODO: this might be a bug in dolfinx: doc claims None and MPI should have same
            #       behavior, None does not work atm.
            spaces = dolfinx.fem.extract_function_spaces(self.F_form)
            self.F = dolfinx.fem.petsc.create_vector(spaces, kind=PETSc.Vec.Type.MPI)
            self.x = self.F.copy()
            self.x.setAttr("_blocks", self.F.getAttr("_blocks"))

            self.x0 = self.F.copy()
            self.x0.setAttr("_blocks", self.F.getAttr("_blocks"))

            if self.restriction is not None:
                # Need to create new global matrix for the restriction
                self._J = dolfinx.fem.petsc.create_matrix(self.J_form)
                self._J.assemble()

                self._x = self.x.copy()

                self.rJ = self.restriction.restrict_matrix(self._J)
                self.rF = self.restriction.restrict_vector(self.F)
                self.rx = self.restriction.restrict_vector(self._x)

                self.snes.setFunction(self._F_block, self.rF)
                self.snes.setJacobian(self._J_block, self.rJ)
            else:
                self.snes.setFunction(self._F_block, self.F)
                self.snes.setJacobian(self._J_block, self.J)

        self.snes.setMonitor(self._monitor_snes)
        self.snes.setConvergenceTest(self._converged)
        self.snes.setOptionsPrefix(prefix)
        self.snes.setFromOptions()

        # Set "active" vectors (= effective dofs)
        if self.restriction is not None:
            self.active_F = self.rF
            self.active_x = self.rx
        else:
            self.active_F = self.F
            self.active_x = self.x0

        # Default monitoring verbosity
        self.verbose = dict(snes=True, ksp=True)

    def _update_functions(self, x: PETSc.Vec) -> None:
        """Update solution functions from the stored vector x."""
        if self.restriction is not None:
            self.restriction.assign(x, [self.u[idx] for idx in self.global_spaces_id])
            dolfinx.fem.petsc.assign([self.u[idx] for idx in self.global_spaces_id], self.x)
        else:
            dolfinx.fem.petsc.assign(x, [self.u[idx] for idx in self.global_spaces_id])  # type: ignore

            for idx in self.global_spaces_id:
                self.u[idx].x.scatter_forward()

            x.copy(self.x)
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

    def _F_block(self, snes: PETSc.SNES, x: PETSc.Vec, F: PETSc.Vec) -> None:
        with self.F.localForm() as f_local:
            f_local.set(0.0)

        self._update_functions(x)

        if self.localsolver is not None:
            self.localsolver.local_update(self)

        dolfinx.fem.petsc.assemble_vector(
            self.F,
            self.F_form,
        )
        dolfinx.fem.petsc.apply_lifting(
            self.F,
            self.J_form,
            bcs=dolfinx.fem.bcs_by_block(
                dolfinx.fem.extract_function_spaces(self.J_form, 1),
                self.bcs,
            ),
            x0=self.x,  # type: ignore
            alpha=-1.0,
        )
        self.F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        dolfinx.fem.petsc.set_bc(
            self.F,
            dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(self.F_form), self.bcs),
            x0=self.x,
            alpha=-1.0,
        )

        if self.restriction is not None:
            self.restriction.restrict_vector(self.F).copy(self.rF)
            self.rF.copy(F)
        else:
            self.F.copy(F)

    def _F_nest(self, snes: PETSc.SNES, x: PETSc.Vec, F: PETSc.Vec) -> None:
        dolfinx.fem.petsc.assign(x, self.u)  # type: ignore
        for u in self.u:
            u.x.scatter_forward()
        x_sub = x.getNestSubVecs()

        bcs1 = dolfinx.fem.bcs.bcs_by_block(
            dolfinx.fem.forms.extract_function_spaces(self.J_form, 1),
            self.bcs,
        )
        for L, F_sub, a in zip(self.F_form, F.getNestSubVecs(), self.J_form):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            dolfinx.fem.petsc.assemble_vector(F_sub, L)
            dolfinx.fem.petsc.apply_lifting(F_sub, a, bcs1, x0=x_sub, alpha=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore

        # Set bc value in RHS
        bcs0 = dolfinx.fem.bcs.bcs_by_block(
            dolfinx.fem.forms.extract_function_spaces(self.F_form), self.bcs
        )
        for F_sub, bc, u_sub in zip(F.getNestSubVecs(), bcs0, x_sub):
            dolfinx.fem.petsc.set_bc(F_sub, bc, u_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def _J_block(self, snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat) -> None:
        self.J.zeroEntries()
        self._update_functions(x)

        dolfinx.fem.petsc.assemble_matrix(self.J, self.J_form, self.bcs, diag=1.0)
        self.J.assemble()

        if self.restriction is not None:
            self.restriction.restrict_matrix(self.J).copy(self.rJ)

    def _J_nest(self, snes: PETSc.SNES, x: PETSc.Vec, J: PETSc.Mat, P: PETSc.Mat):
        self.J.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.J, self.J_form, self.bcs, diag=1.0)
        self.J.assemble()

    def _converged(self, snes, it, norms):
        atol_r = []
        rtol_r = []
        stol_dx = []

        if self.nest:
            self.compute_norms_nest(self.snes)
        else:
            self.compute_norms_block(self.snes)

        for gi, _ in enumerate(self.global_spaces_id):
            atol_r.append(self.norm_r[it][gi] < snes.atol)

            # In some cases, 0th residual of a subfield could be 0.0
            # which would blow relative residual norm
            rtol_r0 = self.norm_r[0][gi]
            if np.isclose(rtol_r0, 0.0):
                rtol_r0 = 1.0

            rtol_r.append(self.norm_r[it][gi] < rtol_r0 * snes.rtol)
            stol_dx.append(self.norm_dx[it][gi] < self.norm_x[it][gi] * snes.stol)

        if it > snes.max_it:
            # https://petsc.org/release/manualpages/SNES/SNES_DIVERGED_MAX_IT/
            return PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT
        elif all(atol_r) and it > 0:
            # https://petsc.org/release/manualpages/SNES/SNES_CONVERGED_FNORM_ABS/
            return PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS
        elif all(rtol_r):
            # https://petsc.org/release/manualpages/SNES/SNES_CONVERGED_FNORM_RELATIVE/
            return PETSc.SNES.ConvergedReason.CONVERGED_FNORM_RELATIVE
        elif all(stol_dx) and it > 0:
            # https://petsc.org/release/manualpages/SNES/SNES_CONVERGED_SNORM_RELATIVE/
            return PETSc.SNES.ConvergedReason.CONVERGED_SNORM_RELATIVE
        else:
            return PETSc.SNES.ConvergedReason.ITERATING

    def _info_ksp(self, ksp: PETSc.KSP) -> str:
        ksp_info = ""

        if ksp.reason < 0:  # type: ignore[operator]
            ksp_info += ANSI.bright_red
            ksp_info += f"failure = {SNESProblem.reasons_ksp[ksp.reason]:s}"
            ksp_info += ANSI.reset

        return ksp_info

    def _info_snes(self, snes: PETSc.SNES) -> str:
        snes_info = ""

        if snes.reason < 0:  # type: ignore[operator]
            snes_info += ANSI.red
            snes_info += f"failure = {SNESProblem.reasons_snes[snes.reason]:s}"
            snes_info += ANSI.reset
        elif snes.reason > 0:  # type: ignore[operator]
            snes_info = ANSI.green
            snes_info += f"success = {SNESProblem.reasons_snes[snes.reason]:s}"
            snes_info += ANSI.reset

        return snes_info

    def _monitor_ksp(self, ksp: PETSc.KSP, ksp_it: int, ksp_norm: float) -> None:
        ksp_info = self._info_ksp(ksp)

        snes_it = self.snes.getIterationNumber()

        message = str(ANSI.bright_black)
        message += f"# SNES iteration {snes_it:2d}, KSP iteration {ksp_it:3d}       |r|={ksp_norm:9.3e} {ksp_info:s}"  # noqa: E501
        message += ANSI.reset

        logger.info(message) if self.verbose["ksp"] else None

    def _monitor_snes(self, snes: PETSc.SNES, snes_it: int, snes_norm: float) -> None:
        snes_info = self._info_snes(snes)

        message = f"# SNES iteration {snes_it:2d} {snes_info:s}"

        logger.info(message) if self.verbose["snes"] else None

        if self.nest:
            self.compute_norms_nest(snes)
        else:
            self.compute_norms_block(snes)

        states = [self.size_x, self.norm_x, self.norm_dx, self.norm_r]

        for gi, i in enumerate(self.global_spaces_id):
            s, x, dx, r = (w[snes_it][gi] for w in states)
            name = self.u[i].name

            message = f"# sub {gi:2d} [{prefixify(s):s}] |x|={x:9.3e} |dx|={dx:9.3e} |r|={r:9.3e} ({name:s})"  # noqa: E501
            logger.info(message) if self.verbose["snes"] else None

        _, x, dx, r = (np.linalg.norm(v) for v in [w[snes_it] for w in states])

        message = f"# all           |x|={x:9.3e} |dx|={dx:9.3e} |r|={r:9.3e}"
        logger.info(message) if self.verbose["snes"] else None

    def status(
        self, verbose: bool = False, error_on_failure: bool = False
    ) -> PETSc.SNES.ConvergedReason:
        if self.snes.getKSP().reason < 0:  # type: ignore[operator]
            if verbose:
                ksp_info = self._info_ksp(self.snes.getKSP())
                message = f"# SNES -> KSP {ksp_info:s}"
                logger.info(message)

            if error_on_failure:
                raise RuntimeError("Linear solver failed!")

        if self.snes.reason < 0:  # type: ignore[operator]
            if verbose:
                snes_info = self._info_snes(self.snes)
                message = f"# SNES {snes_info:s}"
                logger.info(message)

            if error_on_failure:
                raise RuntimeError("Nonlinear solver failed!")

        return self.snes.reason

    def compute_norms_block(self, snes: PETSc.SNES) -> None:
        r = snes.getFunction()[0].getArray(readonly=True)  # type: ignore[index]
        dx = snes.getSolutionUpdate().getArray(readonly=True)
        x = snes.getSolution().getArray(readonly=True)

        # Error per space
        ei_r = np.zeros_like(self.global_spaces_id, dtype=np.float64)
        ei_dx = np.zeros_like(self.global_spaces_id, dtype=np.float64)
        ei_x = np.zeros_like(self.global_spaces_id, dtype=np.float64)
        # Size per space
        si_x = np.zeros_like(self.global_spaces_id, dtype=np.int64)

        offset = 0

        for i in range(len(self.global_spaces_id)):
            if self.restriction is not None:
                # In the restriction case local size if number of
                # owned restricted dofs
                size_local = self.restriction.bglobal_dofs_vec[i].shape[0]
            else:
                size_local = self.u[i].x.petsc_vec.getLocalSize()

            subvec_r = r[offset : offset + size_local]
            subvec_dx = dx[offset : offset + size_local]
            subvec_x = x[offset : offset + size_local]

            # Need first apply square, only then sum over processes
            # i.e. norm is not a linear function
            ei_r[i] = np.sqrt(self.comm.allreduce(np.linalg.norm(subvec_r) ** 2, op=MPI.SUM))
            ei_dx[i] = np.sqrt(self.comm.allreduce(np.linalg.norm(subvec_dx) ** 2, op=MPI.SUM))
            ei_x[i] = np.sqrt(self.comm.allreduce(np.linalg.norm(subvec_x) ** 2, op=MPI.SUM))

            # Store effective global size as sum of locals
            si_x[i] = self.comm.allreduce(size_local, op=MPI.SUM)

            offset += size_local

        it = snes.getIterationNumber()
        self.norm_r[it] = ei_r
        self.norm_dx[it] = ei_dx
        self.norm_x[it] = ei_x
        self.size_x[it] = si_x

    def compute_norms_nest(self, snes: PETSc.SNES) -> None:
        r = snes.getFunction()[0].getNestSubVecs()  # type: ignore
        dx = snes.getSolutionUpdate().getNestSubVecs()
        x = snes.getSolution().getNestSubVecs()

        # Error per space
        ei_r = np.zeros_like(self.global_spaces_id, dtype=np.float64)
        ei_dx = np.zeros_like(self.global_spaces_id, dtype=np.float64)
        ei_x = np.zeros_like(self.global_spaces_id, dtype=np.float64)
        # Size per space
        si_x = np.zeros_like(self.global_spaces_id, dtype=np.int64)

        for i in self.global_spaces_id:
            ei_r[i] = r[i].norm()
            ei_dx[i] = dx[i].norm()
            ei_x[i] = x[i].norm()
            si_x[i] = x[i].getSize()

        it = snes.getIterationNumber()
        self.norm_r[it] = ei_r
        self.norm_dx[it] = ei_dx
        self.norm_x[it] = ei_x
        self.size_x[it] = si_x

    def solve(self, u_init: list[dolfinx.fem.Function] | None = None) -> list[dolfinx.fem.Function]:
        if u_init is not None:
            dolfinx.fem.petsc.assign(u_init, self.x0)
            self.x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore[arg-type]

        self.snes.getKSP().setMonitor(self._monitor_ksp)

        if self.restriction is not None:
            self.snes.solve(None, self.rx)
            self.restriction.assign(self.rx, [self.solution[idx] for idx in self.global_spaces_id])
        else:
            self.snes.solve(None, self.x0)
            dolfinx.fem.petsc.assign(self.x0, [self.solution[idx] for idx in self.global_spaces_id])  # type: ignore[arg-type]

            for idx in self.global_spaces_id:
                self.solution[idx].x.scatter_forward()

        if self.localsolver is not None:
            with self.snes.getSolutionUpdate().localForm() as dx_local:
                dx_local.set(0.0)  # converged solution (fix for single step solves)
            self.localsolver.local_update(self)  # ensure final local update
            dolfinx.fem.petsc.assign(
                self.xloc,  # type: ignore
                [self.solution[idx] for idx in self.localsolver.local_spaces_id],  # type: ignore[arg-type]
            )

            for idx in self.localsolver.local_spaces_id:
                self.solution[idx].x.scatter_forward()

        self.snes.getKSP().cancelMonitor()

        return self.solution

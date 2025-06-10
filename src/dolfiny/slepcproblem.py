import dolfinx
import dolfinx.fem.petsc
import ufl

from slepc4py import SLEPc


class SLEPcProblem:
    def __init__(
        self,
        F_form: list,
        u: list,
        lmbda: dolfinx.fem.Function,
        A_form=None,
        B_form=None,
        prefix=None,
    ):
        """SLEPc problem and solver wrapper.

        Wrapper for a generalised eigenvalue problem obtained from UFL residual forms.

        Parameters
        ----------
        F_form
            Residual forms
        u
            Current solution vectors
        lmbda
            Eigenvalue function. Residual forms must be linear in lmbda for
            linear eigenvalue problems.
        A_form, optional
            Override automatically derived A
        B_form, optional
            Override automatically derived B

        Note
        ----
        In general, eigenvalue problems have form T(lmbda) * x = 0,
        where T(lmbda) is a matrix-valued function.
        Linear eigenvalue problems have T(lmbda) = A + lmbda * B, and if B is not identity matrix
        then this problem is called generalized (linear) eigenvalue problem.

        """
        self.F_form = dolfinx.fem.form(F_form)
        self.u = u
        self.lmbda = lmbda
        self.comm = u[0].function_space.mesh.comm

        self.ur = []
        self.ui = []
        for func in u:
            self.ur.append(dolfinx.fem.Function(func.function_space, name=func.name))
            self.ui.append(dolfinx.fem.Function(func.function_space, name=func.name))

        # Prepare tangent form M0 which has terms involving lambda
        self.M0 = [[None for i in range(len(self.u))] for j in range(len(self.u))]
        for i in range(len(self.u)):
            for j in range(len(self.u)):
                self.M0[i][j] = ufl.algorithms.expand_derivatives(
                    ufl.derivative(
                        F_form[i], self.u[j], ufl.TrialFunction(self.u[j].function_space)
                    )
                )

        if B_form is None:
            B0: list[list[ufl.Form | None]] = [
                [None for i in range(len(self.u))] for j in range(len(self.u))
            ]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    # Differentiate wrt. lambda and replace all remaining lambda with Zero
                    B0[i][j] = ufl.algorithms.expand_derivatives(ufl.diff(self.M0[i][j], lmbda))
                    B0[i][j] = ufl.replace(B0[i][j], {lmbda: ufl.zero()})

                    if B0[i][j].empty():  # type: ignore[union-attr]
                        B0[i][j] = None

            self.B_form = B0
        else:
            self.B_form = B_form

        if A_form is None:
            A0: list[list[ufl.Form | None]] = [
                [None for i in range(len(self.u))] for j in range(len(self.u))
            ]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    A0[i][j] = ufl.replace(self.M0[i][j], {lmbda: ufl.zero()})

                    if A0[i][j].empty():  # type: ignore[union-attr]
                        A0[i][j] = None
                        continue
            self.A_form = A0
        else:
            self.A_form = A_form

        self.eps = SLEPc.EPS().create(self.comm)
        self.eps.setOptionsPrefix(prefix)
        self.eps.setFromOptions()

        self.A_form = dolfinx.fem.form(self.A_form)
        self.B_form = dolfinx.fem.form(self.B_form)

        self.A = dolfinx.fem.petsc.create_matrix(self.A_form)  # type: ignore[arg-type]

        self.B = None
        if not self.empty_B():
            self.B = dolfinx.fem.petsc.create_matrix(self.B_form)  # type: ignore[arg-type]

    def solve(self):
        self.A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.A, self.A_form, [])
        self.A.assemble()

        if not self.empty_B():
            self.B.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(self.B, self.B_form, [])
            self.B.assemble()

        self.eps.setOperators(self.A, self.B)
        self.eps.solve()

    def getEigenpair(self, i):
        xr = dolfinx.fem.petsc.create_vector(self.F_form, kind="mpi")
        xi = xr.duplicate()
        eigval = self.eps.getEigenpair(i, xr, xi)

        dolfinx.fem.petsc.assign(xr, self.ur)
        dolfinx.fem.petsc.assign(xi, self.ui)

        [u.x.scatter_forward() for u in self.ur + self.ui]

        xr.destroy()
        xi.destroy()

        return (eigval, self.ur, self.ui)

    def empty_B(self):
        for i in range(len(self.B_form)):
            for j in range(len(self.B_form[i])):
                if self.B_form[i][j] is not None:
                    return False

        return True

# mypy: disable-error-code="attr-defined, name-defined"

from petsc4py import PETSc

import dolfinx

import numpy as np


class Restriction:
    def __init__(
        self, function_spaces: list[dolfinx.fem.FunctionSpace], blocal_dofs: list[np.ndarray]
    ):
        """Restriction of a problem to subset of degree-of-freedom indices.

        Parameters
        ----------
        function_spaces
            List of function spaces on which restriction operates
        blocal_dofs
            Block-local DOF indices.
        comm: optional

        Note
        ----
        Currently, restriction of a matrix and vector is sub-optimal, since it assumes
        different parallel layout every time restriction is called.

        """
        self.function_spaces = function_spaces
        self.blocal_dofs = blocal_dofs
        self.comm = self.function_spaces[0].mesh.comm

        self.bglobal_dofs_vec = []
        self.bglobal_dofs_mat = []

        self.boffsets_mat = [0]
        self.boffsets_vec = [0]
        offset_mat = 0
        offset_vec = 0

        for i, space in enumerate(function_spaces):
            bs = space.dofmap.index_map_bs

            size_local = space.dofmap.index_map.size_local
            num_ghosts = space.dofmap.index_map.num_ghosts

            self.boffsets_mat.append(self.boffsets_mat[-1] + bs * (size_local + num_ghosts))
            offset_mat += self.boffsets_mat[-1]

            self.boffsets_vec.append(self.boffsets_vec[-1] + bs * size_local)
            offset_vec += self.boffsets_vec[-1]

            dofs = self.blocal_dofs[i].copy()
            # Remove any ghost dofs
            dofs = dofs[dofs < bs * size_local]
            dofs += self.boffsets_mat[i]
            self.bglobal_dofs_mat.append(dofs)

            dofs = self.blocal_dofs[i].copy()
            dofs = dofs[dofs < bs * size_local]
            dofs += self.boffsets_vec[i]
            self.bglobal_dofs_vec.append(dofs)

        self.bglobal_dofs_vec_stacked = np.hstack(self.bglobal_dofs_vec)
        self.bglobal_dofs_mat_stacked = np.hstack(self.bglobal_dofs_mat)

    def restrict_matrix(self, A: PETSc.Mat):
        # Fetching IS only for owned dofs
        # Ghost dofs would get the same global index which would result in
        # duplicate global indices in global IS
        local_isrow = PETSc.IS(self.comm).createGeneral(self.bglobal_dofs_mat_stacked)
        global_isrow = A.getLGMap()[0].applyIS(local_isrow)

        subA = A.createSubMatrix(isrow=global_isrow, iscol=global_isrow)
        subA.assemble()

        return subA

    def restrict_vector(self, x: PETSc.Vec):
        arr = x.array[self.bglobal_dofs_vec_stacked]
        subx = PETSc.Vec().createGhostWithArray([], arr)

        return subx

    def assign(self, rx: PETSc.Vec, f: list):
        """Update Functions using restricted DOF indices."""
        rdof_offset = 0
        for i, fi in enumerate(f):
            num_rdofs = self.bglobal_dofs_vec[i].shape[0]

            fi.x.array[self.bglobal_dofs_vec[i] - self.boffsets_vec[i]] = rx.array_r[
                rdof_offset : (rdof_offset + num_rdofs)
            ]

            fi.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            rdof_offset += num_rdofs

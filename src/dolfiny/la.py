from petsc4py import PETSc

from scipy.sparse import csr_matrix


def petsc_to_scipy(A):
    """Convert PETSc serial matrix to SciPy CSR matrix."""
    ai, aj, av = A.getValuesCSR()
    mat = csr_matrix((av, aj, ai))

    return mat


def scipy_to_petsc(A):
    """Convert SciPy CSR matrix to PETSc serial matrix."""
    nrows = A.shape[0]
    ncols = A.shape[1]

    ai, aj, av = A.indptr, A.indices, A.data
    mat = PETSc.Mat()
    mat.createAIJ(size=(nrows, ncols))
    mat.setUp()
    mat.setValuesCSR(ai, aj, av)
    mat.assemble()

    return mat


def is_symmetric(A, rtol=1e-06, atol=1e-08, normtype=PETSc.NormType.INFINITY):
    """Test for symmetry of operator A."""
    assert isinstance(A, PETSc.Mat)

    knows, fact = A.isSymmetricKnown()

    if knows and fact:
        return True
    else:
        asymA = 0.5 * (A - PETSc.Mat().createTranspose(A))
        norm_asymA = asymA.norm(normtype)
        asymA.destroy()

        import dolfiny

        dolfiny.utils.pprint(f"absolute asymmetry measure = {norm_asymA:.3e}")
        dolfiny.utils.pprint(f"relative asymmetry measure = {norm_asymA / A.norm(normtype):.3e}")

        return norm_asymA < atol or norm_asymA / A.norm(normtype) < rtol

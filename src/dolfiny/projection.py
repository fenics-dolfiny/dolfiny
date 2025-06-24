from petsc4py import PETSc

import dolfinx
import ufl
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc


def project(e, target_func, bcs=[]):
    """Project UFL expression.

    Note:
    ----
    This method solves a linear system (using KSP defaults).

    """
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    v = ufl.TrialFunction(V)
    a = dolfinx.fem.form(ufl.inner(v, w) * dx)
    L = dolfinx.fem.form(ufl.inner(e, w) * dx)

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setType("bcgs")
    solver.getPC().setType("bjacobi")
    solver.rtol = 1.0e-05
    solver.setOperators(A)
    solver.solve(b, target_func.x.petsc_vec)
    assert solver.reason > 0
    target_func.x.scatter_forward()

    # Destroy PETSc linear algebra objects and solver
    solver.destroy()
    A.destroy()
    b.destroy()


def project_codimension(p_expression, target_func, projector, mt, mt_id, eps=1.0e-03):
    """Project expression defined on codimension (on a mesh-tagged subset) into given function.

    Args:
        p_expression: The (projected) expression.
        target_func: The target function.
        projector: The functor that performs the expected projection.
        mt: Meshtags
        mt_id: Meshtag id that determines the set of codimension entities.
        eps: Augmentation factor.

    """
    import numpy as np

    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    ds = ufl.ds(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    v = ufl.TrialFunction(V)
    ε = eps  # * ufl.FacetArea(V.mesh)
    a = dolfinx.fem.form(ufl.inner(projector(v), projector(w)) * ds + ε * ufl.inner(v, w) * ds)
    L = dolfinx.fem.form(ufl.inner(p_expression, projector(w)) * ds)

    # Get dofs not associated with mt = inactive
    dofs_mt = dolfinx.fem.locate_dofs_topological(V, V.mesh.topology.dim - 1, mt.find(mt_id))
    dofsall = np.arange(
        V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, dtype=np.int32
    )
    dofs_inactive = np.setdiff1d(dofsall, dofs_mt, assume_unique=True)

    # Zero-valued inactive dofs
    zero = dolfinx.fem.Function(V)
    bcs = [dolfinx.fem.dirichletbc(zero, dofs_inactive)]

    # Create operator
    pattern = dolfinx.fem.create_sparsity_pattern(a)
    pattern.insert_diagonal(dofs_inactive)
    pattern.finalize()
    A = dolfinx.cpp.la.petsc.create_matrix(V.mesh.comm, pattern)

    # Assemble linear system
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix_mat(A, a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setType("bcgs")
    solver.getPC().setType("bjacobi")
    solver.rtol = eps * 1.0e-02
    solver.setOperators(A)
    solver.solve(b, target_func.x.petsc_vec)
    assert solver.reason > 0
    target_func.x.scatter_forward()

    # Destroy PETSc linear algebra objects and solver
    solver.destroy()
    A.destroy()
    b.destroy()

    return target_func

from collections.abc import Sequence

from petsc4py import PETSc

import dolfinx
import ufl
import ufl.form
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

    # Establish function space and mesh tags
    V = target_func.function_space
    w = ufl.TestFunction(V)
    v = ufl.TrialFunction(V)

    # Transform all input to lists if not already
    if not isinstance(p_expression, Sequence):
        p_expression = [p_expression]
    if not isinstance(projector, Sequence):
        projector = [projector]
    if not isinstance(mt, Sequence):
        mt = [mt]
    if not isinstance(mt_id, Sequence):
        mt_id = [mt_id]

    # Assert all input lists have the same length
    assert len(p_expression) == len(projector) == len(mt) == len(mt_id), (
        "All input lists must have the same length."
    )

    # Loop over all input expressions and add to projection

    a = ufl.form.Zero()
    L = ufl.form.Zero()
    cells = np.empty(0, dtype=np.int32)

    # Ensure we have a mesh and attach to measure
    for p_expression_k, projector_k, mt_k, mt_id_k in zip(p_expression, projector, mt, mt_id):
        # Ensure we have a mesh and attach to measure
        ds = ufl.ds(domain=V.mesh, subdomain_data=mt_k, subdomain_id=mt_id_k)

        # Add to variational problem for projection
        ε = eps  # * ufl.FacetArea(V.mesh)
        a += ufl.inner(projector_k(v), projector_k(w)) * ds + ε * ufl.inner(v, w) * ds
        L += ufl.inner(p_expression_k, projector_k(w)) * ds

        # Cells in the mesh tags
        cells = np.concatenate((cells, mt_k.find(mt_id_k)), axis=0)

    # Create variational forms
    a = dolfinx.fem.form(a)
    L = dolfinx.fem.form(L)

    # Get dofs not associated with mt = inactive
    dofs_mt = dolfinx.fem.locate_dofs_topological(V, V.mesh.topology.dim - 1, cells)
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
    dolfinx.fem.petsc.assemble_matrix(A, a, bcs)
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

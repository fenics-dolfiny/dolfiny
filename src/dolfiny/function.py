# mypy: disable-error-code="attr-defined"

import dolfinx
import ufl

import numpy as np


def extract_blocks(
    form, test_functions: list[ufl.Argument], trial_functions: list[ufl.Argument] | None = None
):
    """Extract blocks from a monolithic UFL form.

    Parameters
    ----------
    form
    test_functions
    trial_functions: optional

    Returns
    -------
    Split UFL form in the order determined by the passed test and trial functions.
    If no `trial_functions` are provided returns a list, otherwise returns list of lists.

    """
    # Check for distinct test functions
    if len(test_functions) != len(set(test_functions)):
        raise RuntimeError(
            "Duplicate test functions detected. Create TestFunctions from separate FunctionSpaces!"
        )

    # Prepare empty block matrices list
    if trial_functions is not None:
        blocks: list[list[None]] = [[None] * len(test_functions)] * len(trial_functions)
    else:
        blocks: list[None] = [None] * len(test_functions)  # type: ignore[no-redef]

    for i, tef in enumerate(test_functions):
        if trial_functions is not None:
            for j, trf in enumerate(trial_functions):
                to_null = dict()

                # Dictionary mapping the other trial functions
                # to zero
                for item in trial_functions:
                    if item != trf:
                        to_null[item] = ufl.zero(item.ufl_shape)

                # Dictionary mapping the other test functions
                # to zero
                for item in test_functions:
                    if item != tef:
                        to_null[item] = ufl.zero(item.ufl_shape)

                blocks[i][j] = ufl.replace(form, to_null)
        else:
            to_null = dict()

            # Dictionary mapping the other test functions
            # to zero
            for item in test_functions:
                if item != tef:
                    to_null[item] = ufl.zero(item.ufl_shape)

            blocks[i] = ufl.replace(form, to_null)

    return blocks


def unroll_dofs(dofs, block_size):
    """Unroll blocked dofs."""
    arr = block_size * np.repeat(dofs, block_size).reshape(-1, block_size) + np.arange(block_size)
    return arr.flatten().astype(dofs.dtype)


def evaluate(f: dolfinx.fem.Function, x: np.ndarray, tdim: int | None = None) -> np.ndarray:
    """Evaluate function f at points x (along a polygonal path).

    Parameters
    ----------
    f : dolfinx.fem.Function
        The function to evaluate, defined on a finite element function space.
    x : np.ndarray
        An array of shape (3, n) containing the coordinates of points (along the path).
    tdim : int, optional
        The topological dimension of the entities to consider. If not provided, it
        defaults to the topological dimension of the mesh.

    Returns
    -------
    np.ndarray
        An array of evaluated function values at the specified points.

    Raises
    ------
    ValueError
        If the shape of x is not (3, n).

    """
    if x.ndim != 2 or x.shape[0] != 3:
        raise ValueError(f"Path coordinates x must have shape (3, n); received shape: {x.shape}.")

    # Get mesh and its topological dimension
    mesh = f.function_space.mesh
    if tdim is None:
        tdim = mesh.topology.dim

    # Create a bounding box tree for spatial queries
    bounding_box_tree = dolfinx.geometry.bb_tree(mesh, tdim)

    # Find candidate cells for the provided points
    cell_candidates = dolfinx.geometry.compute_collisions_points(bounding_box_tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)

    # Identify owned cells and compute owned values
    owned_indices = np.where(np.diff(colliding_cells.offsets) > 0)[0]
    owned_cells = colliding_cells.array[colliding_cells.offsets[owned_indices]]
    owned_values = f.eval(x[:, owned_indices].T, owned_cells)

    # Gather owned cell indices and evaluated values across all ranks
    gathered_indices = mesh.comm.allgather(owned_indices)
    gathered_values = mesh.comm.allgather(owned_values)

    # Prepare an output array for the evaluated function values
    evaluated_function_values = np.empty((x.shape[1], f.function_space.value_size), dtype=f.dtype)

    # Populate the output array with gathered values
    all_indices = np.concatenate(gathered_indices)
    all_values = np.concatenate(gathered_values)
    evaluated_function_values[all_indices] = all_values

    return evaluated_function_values

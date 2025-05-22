# mypy: disable-error-code="attr-defined"

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

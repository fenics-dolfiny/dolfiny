from mpi4py import MPI

import dolfinx
import ufl

import pytest

import dolfiny.ufl_utils as ufl_utils


def test_visualize_form_uses_graph_backend(tmp_path):
    pytest.importorskip("pygraphviz")

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    form = u * v * ufl.dx(domain=mesh)

    ufl_utils.visualize(form, tmp_path / "form_graph.png")

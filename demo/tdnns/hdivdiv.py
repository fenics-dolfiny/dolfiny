#!/usr/bin/env python3

import warnings

import basix
import basix.ufl

import numpy as np

warnings.filterwarnings("error")


def topological_dimension(celltype):
    if celltype == basix.CellType.interval:
        return 1
    if celltype == basix.CellType.triangle:
        return 2
    if celltype == basix.CellType.tetrahedron:
        return 3


def polyset_dim(celltype, d):
    if celltype == basix.CellType.interval:
        return d + 1
    if celltype == basix.CellType.triangle:
        return (d + 1) * (d + 2) // 2
    if celltype == basix.CellType.tetrahedron:
        return (d + 1) * (d + 2) * (d + 3) // 6


def sub_entity_type(celltype, dim, index):
    tdim = topological_dimension(celltype)
    assert dim >= 0 and dim <= tdim
    if dim == 0:
        return basix.CellType.point
    elif dim == 1:
        return basix.CellType.interval
    elif dim == tdim:
        return celltype
    emap = {3: basix.CellType.triangle, 4: basix.CellType.quadrilateral}
    topo = basix._basixcpp.topology(celltype._value_)
    return emap[len(topo[dim][index])]


def discrete(celltype, quadrature_degree, entity_degree, variant=basix.LagrangeVariant.legendre):
    numdofs = polyset_dim(celltype, entity_degree)

    if quadrature_degree >= 0:
        qa, qw = basix.make_quadrature(celltype, quadrature_degree)
    else:
        qa, qw = np.zeros((0, topological_dimension(celltype))), np.zeros(0)

    if entity_degree >= 0:
        space = basix.create_element(
            basix.ElementFamily.P, celltype, entity_degree, variant, discontinuous=True
        )
        values = space.tabulate(0, qa)
    else:
        values = None

    return numdofs, qa, qw, values


def points_entity(qa, e_x):
    x = np.zeros((qa.shape[0], e_x.shape[1]))
    x[:] = e_x[0]
    for k in range(e_x.shape[0] - 1):
        x[:] += np.outer(qa[:, k], e_x[k + 1] - e_x[0])
    return x


def moments_entity(numdofs, tdim, moment_values, qw, directions):
    ndirections = len(directions)
    nquadrature = 0 if moment_values is None else moment_values.shape[1]
    M = np.zeros((numdofs * ndirections, tdim * tdim, nquadrature, 1))
    for n in range(numdofs):
        for j, P in enumerate(directions):
            M[n * ndirections + j, :, :, 0] = np.outer(P.flatten(), qw * moment_values[0, :, n, 0])
    return M


def orthogonalise(A, start=0):
    """In-place row-orthogonalisation of matrix A."""
    assert len(A.shape) == 2
    for i, Ai in enumerate(A[start:]):
        Ai -= A[start:i] @ Ai @ A[start:i]
        norm = np.linalg.norm(Ai)
        assert norm > 4 * np.finfo(A.dtype).eps
        Ai /= norm
    assert np.allclose(np.linalg.norm(A, axis=1), 1.0)
    return A


def create_custom_hdivdiv(celltype, degree, discontinuous=False, verbose=False):
    assert degree >= 0
    assert celltype in (basix.CellType.triangle, basix.CellType.tetrahedron)

    tdim = topological_dimension(celltype)

    nc = tdim * (tdim + 1) // 2  # number of distinct (tensor) components [symmetry]
    basis_size = polyset_dim(celltype, degree)
    ndofs = basis_size * nc
    psize = basis_size * tdim * tdim  # unrolled tensor representation

    if verbose:
        print(f"[Hdivdiv] ctype = {celltype}, degree = {degree:2d}, ndofs = {ndofs:3d}")

    wcoeffs = np.zeros((ndofs, psize))

    for i in range(tdim):
        for j in range(tdim):
            xoff = i + tdim * j
            yoff = i + j
            if tdim == 3 and i > 0 and j > 0:
                yoff += 1

            for k in range(basis_size):
                wcoeffs[yoff * basis_size + k, xoff * basis_size + k] = (
                    1.0 if i == j else np.sqrt(0.5)
                )

    topology = basix.topology(celltype)

    #    n   e   f   v   := nodes, edges, faces, volumes
    x = [[], [], [], []]
    M = [[], [], [], []]

    # Loop over entities
    for d, entities in enumerate(topology):
        # Loop over entities of dimension d
        for e, _ in enumerate(entities):
            ex = basix._basixcpp.sub_entity_geometry(celltype._value_, d, e)
            ct = sub_entity_type(celltype, d, e)

            if tdim - d > 1:  # entity not relevant
                x[d].append(np.zeros((0, tdim)))
                M[d].append(np.zeros((0, tdim * tdim, 0, 1)))

            if tdim - d == 1:  # facets with NormalInnerProductIntegralMoment
                K = ex @ basix._basixcpp.cell_facet_jacobians(celltype._value_)[e]

                facet_n = basix._basixcpp.cell_facet_outward_normals(celltype._value_)[
                    e
                ]  # normalised
                facet_n *= np.sqrt(np.linalg.det(K.T @ K))

                directions = [np.outer(facet_n, facet_n)]

                numdofs, qa, qw, moment_values = discrete(ct, 2 * degree, degree)

                x[d].append(points_entity(qa, ex))
                M[d].append(moments_entity(numdofs, tdim, moment_values, qw, directions))

            if tdim - d == 0:  # cells with IntegralMoment
                if ct == basix.CellType.triangle:
                    # see A. Sinwel 2009, PhD thesis
                    directions = [
                        np.array([[0.0, 1.0], [1.0, 0.0]]),
                        np.array([[-2.0, 1.0], [1.0, 0.0]]),
                        np.array([[0.0, -1.0], [-1.0, 2.0]]),
                    ]
                    directions_extra = []
                if ct == basix.CellType.tetrahedron:
                    # see A. Pechstein, J. Schöberl 2018, https://doi.org/10.1007/s00211-017-0933-3
                    directions = [
                        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
                        np.array([[-6.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
                        np.array([[0.0, 1.0, 1.0], [1.0, -6.0, 1.0], [1.0, 1.0, 0.0]]),
                        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, -6.0]]),
                    ]
                    directions_extra = [
                        np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [-1.0, 1.0, 0.0]]),
                        np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                    ]

                # checks
                # ruff: noqa: E501
                # orth_fd_cd = [np.einsum('ij,ij', Si, Sj) for Si in directions for Sj in directions_extra]
                # assert np.isclose(orth_fd_cd, 0.0).all()
                # sidx = [0, 1, 3] if tdim == 2 else [0, 1, 2, 4, 5, 8]
                # basis = np.array(directions + directions_extra).flatten().reshape(3 * (tdim - 1), -1)[:, sidx]
                # assert abs(np.linalg.det(basis)) > 0.0
                # fnormals = basix.cell.facet_normals(ct)
                # nf = len(fnormals)
                # K = np.array([n.T @ Si @ n for Si in directions for n in fnormals]).reshape(nf, -1)
                # assert np.allclose(K, np.diag(np.diag(K)))
                # K = np.array([n.T @ Si @ n for Si in directions_extra for n in fnormals]).reshape(nf, -1)
                # assert np.allclose(K, np.zeros_like(K))

                # basix HHJ triangle uses quadrature_degree = 2 * degree - 1
                quad_degree = 2 * degree - 1 if ct == basix.CellType.triangle else 2 * degree
                numdofs_b, qa, qw, moment_values_b = discrete(ct, quad_degree, degree - 1)
                numdofs_e, __, __, moment_values_e = discrete(ct, quad_degree, degree - 0)

                M_b = moments_entity(numdofs_b, tdim, moment_values_b, qw, directions)
                M_e = moments_entity(numdofs_e, tdim, moment_values_e, qw, directions_extra)

                x[d] = [points_entity(qa, ex)]
                M[d] = [M_e if moment_values_b is None else np.concatenate([M_b, M_e], axis=0)]

        if verbose:
            print(d, [m.shape for m in M[d]])
            # print(d, [z.shape for z in x[d]])

    if discontinuous:
        # x, M = make_discontinuous(x, M)
        raise RuntimeError("Flag `discontinuous=True` not supported!")

    space = basix.SobolevSpace.L2 if discontinuous else basix.SobolevSpace.HDivDiv
    map_type = basix.MapType.doubleContravariantPiola

    # Create basix element
    hdivdiv = basix.ufl.custom_element(
        celltype, [tdim, tdim], wcoeffs, x, M, 0, map_type, space, discontinuous, -1, degree
    )

    return hdivdiv


def test_hdivdiv_zero_normal_normal_cell(element):
    if isinstance(element, basix.ufl._BasixElement):
        element = element._element

    ndofs = element.dim
    degree = element.degree
    tdim = topological_dimension(element.cell_type)

    topology = basix._basixcpp.topology(element.cell_type._value_)

    if element.cell_type == basix.CellType.triangle:
        num_cell_dofs = 3 * (degree + 1) * degree // 2
    elif element.cell_type == basix.CellType.tetrahedron:
        num_cell_dofs = (degree + 2) * (degree + 1) ** 2
    else:
        raise RuntimeError("Unexpected cell type!")

    if num_cell_dofs == 0:
        return

    normal_normal = np.zeros(ndofs)

    # Check for vanishing facet normal-normal components of cell functions
    for e, _ in enumerate(topology[tdim - 1]):
        ex = basix._basixcpp.sub_entity_geometry(element.cell_type._value_, tdim - 1, e)
        ct = sub_entity_type(element.cell_type, tdim - 1, e)
        qa, qw = basix.make_quadrature(ct, 8)
        p = points_entity(qa, ex)
        n = basix.cell.facet_normals(element.cell_type)[e]  # normalised
        t = element.tabulate(0, p)
        for k, w in enumerate(qw):
            for dof in range(ndofs):
                M = t[0, k, dof].reshape(tdim, tdim)
                normal_normal[dof] += abs(n.T @ M @ n * w)

    assert np.allclose(normal_normal[-num_cell_dofs:], 0.0)


if __name__ == "__main__":
    for ctype in [basix.CellType.triangle, basix.CellType.tetrahedron]:
        for degree in range(5):
            hdivdiv = create_custom_hdivdiv(ctype, degree, verbose=True)
            test_hdivdiv_zero_normal_normal_cell(hdivdiv)

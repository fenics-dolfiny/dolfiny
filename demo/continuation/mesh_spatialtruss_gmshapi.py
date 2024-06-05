#!/usr/bin/env python3

from mpi4py import MPI


def mesh_spatialtruss_gmshapi(
    name="spatialtruss",
    shape="straight",
    L=1.0,
    nL=2,
    θ=3.1415 / 3,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of spatial truss using the Python API of Gmsh.
    """

    tdim = 1  # target topological dimension

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        # Add model under given name
        gmsh.model.add(name)

        # Create points and lines
        φ = 3.1415 / 6
        p0 = gmsh.model.geo.addPoint(
            -L * gmsh.numpy.cos(θ) * gmsh.numpy.cos(φ),
            -L * gmsh.numpy.cos(θ) * gmsh.numpy.sin(φ),
            0.0,
        )
        p1 = gmsh.model.geo.addPoint(
            +L * gmsh.numpy.cos(θ) * gmsh.numpy.cos(φ),
            -L * gmsh.numpy.cos(θ) * gmsh.numpy.sin(φ),
            0.0,
        )
        p2 = gmsh.model.geo.addPoint(0.0, +L * gmsh.numpy.cos(θ), 0.0)
        p3 = gmsh.model.geo.addPoint(0.0, 0.0, +L * gmsh.numpy.sin(θ))
        p4 = gmsh.model.geo.addPoint(0.0, 0.0, L + L * gmsh.numpy.sin(θ))

        if shape == "straight":
            l0 = gmsh.model.geo.addLine(p0, p3)
            l1 = gmsh.model.geo.addLine(p1, p3)
            l2 = gmsh.model.geo.addLine(p2, p3)
            l3 = gmsh.model.geo.addLine(p3, p4)
            lines = [l0, l1, l2, l3]
        else:
            raise RuntimeError("Unknown shape identifier '{shape:s}'")

        # Sync
        gmsh.model.geo.synchronize()
        # Define physical groups for subdomains (! target tag > 0)
        lower = 1
        gmsh.model.addPhysicalGroup(tdim, [l0, l1, l2], lower)
        gmsh.model.setPhysicalName(tdim, lower, "lower")
        upper = 2
        gmsh.model.addPhysicalGroup(tdim, [l3], upper)
        gmsh.model.setPhysicalName(tdim, upper, "upper")
        # Define physical groups for interfaces (! target tag > 0)
        support = 1
        gmsh.model.addPhysicalGroup(tdim - 1, [p0, p1, p2], support)
        gmsh.model.setPhysicalName(tdim - 1, support, "support")
        connect = 2
        gmsh.model.addPhysicalGroup(tdim - 1, [p3], connect)
        gmsh.model.setPhysicalName(tdim - 1, connect, "connect")
        verytop = 3
        gmsh.model.addPhysicalGroup(tdim - 1, [p4], verytop)
        gmsh.model.setPhysicalName(tdim - 1, verytop, "verytop")

        # Set refinement along curve direction
        for line in lines:
            gmsh.model.mesh.setTransfiniteCurve(line, numNodes=nL, meshType="Progression", coef=1.0)

        # Generate the mesh
        gmsh.model.mesh.generate()

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    mesh_spatialtruss_gmshapi(msh_file="spatialtruss.msh")

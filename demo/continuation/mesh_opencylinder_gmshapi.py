#!/usr/bin/env python3

from mpi4py import MPI


def mesh_opencylinder_gmshapi(
    name="opencylinder",
    Ly=1.0,
    R=1.0,
    θ=3.1415 / 2,
    nL=10,
    nR=10,
    do_quads=False,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of an open cylinder using the Python API of Gmsh.
    """

    tdim = 2  # target topological dimension

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        if do_quads:
            nL = int(nL / 2) * 2 + 1  # make sure nL is odd
            nR = int(nR / 2) * 2 + 1  # make sure nR is odd
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)

        # Add model under given name
        gmsh.model.add(name)

        # Create points
        p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
        p1 = gmsh.model.geo.addPoint(+R * gmsh.numpy.sin(θ), 0.0, +R * gmsh.numpy.cos(θ))
        p2 = gmsh.model.geo.addPoint(0.0, 0.0, +R)
        p4 = gmsh.model.geo.addPoint(-R * gmsh.numpy.sin(θ), 0.0, +R * gmsh.numpy.cos(θ))

        p5 = gmsh.model.geo.addPoint(0.0, +Ly, 0.0)
        p6 = gmsh.model.geo.addPoint(+R * gmsh.numpy.sin(θ), +Ly, +R * gmsh.numpy.cos(θ))
        p7 = gmsh.model.geo.addPoint(0.0, +Ly, +R)
        p8 = gmsh.model.geo.addPoint(-R * gmsh.numpy.sin(θ), +Ly, +R * gmsh.numpy.cos(θ))

        p9 = gmsh.model.geo.addPoint(0.0, +2 * Ly, 0.0)
        p10 = gmsh.model.geo.addPoint(+R * gmsh.numpy.sin(θ), +2 * Ly, +R * gmsh.numpy.cos(θ))
        p11 = gmsh.model.geo.addPoint(0.0, +2 * Ly, +R)
        p12 = gmsh.model.geo.addPoint(-R * gmsh.numpy.sin(θ), +2 * Ly, +R * gmsh.numpy.cos(θ))

        # Create lines
        l0 = gmsh.model.geo.addCircleArc(p1, p0, p2)
        l1 = gmsh.model.geo.addCircleArc(p4, p0, p2)
        l2 = gmsh.model.geo.addCircleArc(p6, p5, p7)
        l3 = gmsh.model.geo.addCircleArc(p8, p5, p7)
        l4 = gmsh.model.geo.addCircleArc(p10, p9, p11)
        l5 = gmsh.model.geo.addCircleArc(p12, p9, p11)

        l6 = gmsh.model.geo.addLine(p1, p6)
        l7 = gmsh.model.geo.addLine(p6, p10)
        l8 = gmsh.model.geo.addLine(p4, p8)
        l9 = gmsh.model.geo.addLine(p8, p12)
        l10 = gmsh.model.geo.addLine(p2, p7)
        l11 = gmsh.model.geo.addLine(p7, p11)

        # Create loops
        c0 = gmsh.model.geo.addCurveLoop([-l0, l6, l2, -l10])
        c1 = gmsh.model.geo.addCurveLoop([-l2, l7, l4, -l11])
        c2 = gmsh.model.geo.addCurveLoop([l1, l10, -l3, -l8])
        c3 = gmsh.model.geo.addCurveLoop([l3, l11, -l5, -l9])

        # Create surfaces
        s0 = gmsh.model.geo.addSurfaceFilling([c0])
        s1 = gmsh.model.geo.addSurfaceFilling([c1])
        s2 = gmsh.model.geo.addSurfaceFilling([c2])
        s3 = gmsh.model.geo.addSurfaceFilling([c3])
        surfaces = [s0, s1, s2, s3]

        # Sync
        gmsh.model.geo.synchronize()
        # Define physical groups for subdomains (! target tag > 0)
        domain = 1
        gmsh.model.addPhysicalGroup(tdim, surfaces, domain)
        gmsh.model.setPhysicalName(tdim, domain, "domain")
        # Define physical groups for interfaces (! target tag > 0)
        sides = 1
        gmsh.model.addPhysicalGroup(tdim - 1, [l6, l7, l8, l9], sides)
        gmsh.model.setPhysicalName(tdim - 1, sides, "sides")
        front = 2
        gmsh.model.addPhysicalGroup(tdim - 1, [l0, l1], front)
        gmsh.model.setPhysicalName(tdim - 1, front, "front")
        rear = 3
        gmsh.model.addPhysicalGroup(tdim - 1, [l4, l5], rear)
        gmsh.model.setPhysicalName(tdim - 1, rear, "rear")
        top = 4
        gmsh.model.addPhysicalGroup(tdim - 1, [l10, l11], top)
        gmsh.model.setPhysicalName(tdim - 1, top, "top")
        # Define physical groups for co-dimension 2 (! target tag > 0)
        topfront = 1
        gmsh.model.addPhysicalGroup(tdim - 2, [p2], topfront)
        gmsh.model.setPhysicalName(tdim - 2, topfront, "topfront")
        topmid = 2
        gmsh.model.addPhysicalGroup(tdim - 2, [p7], topmid)
        gmsh.model.setPhysicalName(tdim - 2, topmid, "topmid")
        topend = 3
        gmsh.model.addPhysicalGroup(tdim - 2, [p11], topend)
        gmsh.model.setPhysicalName(tdim - 2, topend, "topend")

        # Set refinement along curve direction
        for line in [l6, l7, l8, l9, l10, l11]:
            gmsh.model.mesh.setTransfiniteCurve(line, numNodes=nL, meshType="Progression", coef=1.0)
        for line in [l0, l1, l2, l3, l4, l5]:
            gmsh.model.mesh.setTransfiniteCurve(line, numNodes=nR, meshType="Progression", coef=1.0)

        # Generate the mesh
        gmsh.model.mesh.generate()

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    mesh_opencylinder_gmshapi(msh_file="opencylinder.msh")

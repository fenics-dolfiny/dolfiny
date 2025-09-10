#!/usr/bin/env python3

from mpi4py import MPI


def mesh_block3d_gmshapi(
    name="block3d",
    dx=1.0,
    dy=1.0,
    dz=1.0,
    nx=10,
    ny=10,
    nz=10,
    x0=0.0,
    y0=0.0,
    z0=0.0,
    do_quads=False,
    px=1.0,
    py=1.0,
    pz=1.0,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 3d block using the Python API of Gmsh.
    """
    tdim = 3  # target topological dimension

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        # gmsh.option.setNumber("Mesh.Algorithm", 9)
        # gmsh.option.setNumber("Mesh.Algorithm3D", 10)

        # if do_quads:
        #   gmsh.option.setNumber("Mesh.RecombineAll", 1)
        #   gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
        #   gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)

        # Add model under given name
        gmsh.model.add(name)

        # Set refinement
        import numpy as np

        def nele(n):
            return np.ones(n, dtype=int)

        def prog(n, p):
            t = np.linspace(0, 1, n + 1)[1:]
            if p > 0:
                r = t**p
            else:
                r = 1 - (1 - t) ** (-p)
            return r

        # Create block by extrusion from a point
        p0 = gmsh.model.geo.addPoint(x0, y0, z0)
        # e0 = gmsh.model.geo.extrude([(tdim - 3, p0)], dx, 0.0, 0.0, recombine=do_quads)
        e0 = gmsh.model.geo.extrude(
            [(tdim - 3, p0)],
            dx,
            0.0,
            0.0,
            numElements=nele(nx),
            heights=prog(nx, px),
            recombine=do_quads,
        )
        l0 = e0[1][1]
        # e1 = gmsh.model.geo.extrude([(tdim - 2, l0)], 0.0, dy, 0.0, recombine=do_quads)
        e1 = gmsh.model.geo.extrude(
            [(tdim - 2, l0)],
            0.0,
            dy,
            0.0,
            numElements=nele(ny),
            heights=prog(ny, py),
            recombine=do_quads,
        )
        s0 = e1[1][1]
        # e2 = gmsh.model.geo.extrude([(tdim - 1, s0)], 0.0, 0.0, dz, recombine=do_quads)
        _ = gmsh.model.geo.extrude(
            [(tdim - 1, s0)],
            0.0,
            0.0,
            dz,
            numElements=nele(nz),
            heights=prog(nz, pz),
            recombine=do_quads,
        )

        # Synchronize
        gmsh.model.geo.synchronize()

        # Get model entities
        _points, _lines, surfaces, volumes = (gmsh.model.getEntities(d) for d in [0, 1, 2, 3])

        # Extract surfaces and volumes
        s0, s1, s2, s3, s4, s5 = (
            surfaces[i][1] for i in [5, 0, 1, 3, 4, 2]
        )  # ordering by inspection
        (v0,) = (volumes[i][1] for i in [0])

        # Define physical groups for subdomains (! target tag > 0)
        domain = 1
        gmsh.model.addPhysicalGroup(tdim, [v0], domain)
        gmsh.model.setPhysicalName(tdim, domain, "domain")

        # Define physical groups for interfaces (! target tag > 0)
        surface_front = 1
        gmsh.model.addPhysicalGroup(tdim - 1, [s0], surface_front)
        gmsh.model.setPhysicalName(tdim - 1, surface_front, "surface_front")
        surface_back = 2
        gmsh.model.addPhysicalGroup(tdim - 1, [s1], surface_back)
        gmsh.model.setPhysicalName(tdim - 1, surface_back, "surface_back")
        surface_bottom = 3
        gmsh.model.addPhysicalGroup(tdim - 1, [s2], surface_bottom)
        gmsh.model.setPhysicalName(tdim - 1, surface_bottom, "surface_bottom")
        surface_top = 4
        gmsh.model.addPhysicalGroup(tdim - 1, [s3], surface_top)
        gmsh.model.setPhysicalName(tdim - 1, surface_top, "surface_top")
        surface_left = 5
        gmsh.model.addPhysicalGroup(tdim - 1, [s4], surface_left)
        gmsh.model.setPhysicalName(tdim - 1, surface_left, "surface_left")
        surface_right = 6
        gmsh.model.addPhysicalGroup(tdim - 1, [s5], surface_right)
        gmsh.model.setPhysicalName(tdim - 1, surface_right, "surface_right")

        # Ensure union jack meshing for triangular elements
        # if not do_quads:
        #     gmsh.model.mesh.setTransfiniteSurface(s0, arrangement="Alternate")
        #     gmsh.model.mesh.setTransfiniteSurface(s1, arrangement="Alternate")

        # Generate the mesh
        gmsh.model.mesh.generate()

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    mesh_block3d_gmshapi(msh_file="block3d.msh")

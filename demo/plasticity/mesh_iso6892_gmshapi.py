#!/usr/bin/env python3

from mpi4py import MPI


def mesh_iso6892_gmshapi(
    name="iso6892",
    l0=0.10,
    d0=0.02,
    nr=5,
    x0=0.0,
    y0=0.0,
    z0=0.0,
    do_hexes=False,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 3d tensile test specimen according to ISO 6892-1:2019
    using the Python API of Gmsh.
    """
    tdim = 3  # target topological dimension

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)

        if do_hexes:
            nr = int(nr / 2)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)

        # Add model under given name
        gmsh.model.add(name)

        # Construct using OpenCascade
        grip1 = gmsh.model.occ.addCylinder(x0 - (l0 + 6 * d0) / 2, y0, z0, +2.5 * d0, 0.0, 0.0, d0)
        grip2 = gmsh.model.occ.addCylinder(x0 + (l0 + 6 * d0) / 2, y0, z0, -2.5 * d0, 0.0, 0.0, d0)

        takeout1 = gmsh.model.occ.addTorus(x0 - (l0 + d0) / 2, y0, z0, 1.0 * d0, 0.5 * d0)
        takeout2 = gmsh.model.occ.addTorus(x0 + (l0 + d0) / 2, y0, z0, 1.0 * d0, 0.5 * d0)

        gmsh.model.occ.rotate(
            [(tdim, takeout1)], x0 - (l0 + d0) / 2, y0, z0, 0.0, 1.0, 0.0, gmsh.pi / 2
        )
        gmsh.model.occ.rotate(
            [(tdim, takeout2)], x0 + (l0 + d0) / 2, y0, z0, 0.0, 1.0, 0.0, gmsh.pi / 2
        )

        grip1 = gmsh.model.occ.cut([(tdim, grip1)], [(tdim, takeout1)])
        grip2 = gmsh.model.occ.cut([(tdim, grip2)], [(tdim, takeout2)])

        gmsh.model.occ.addCylinder(x0 - (l0 + d0) / 2, y0, z0, +d0 / 2, 0.0, 0.0, d0 / 2)  # conn1
        gmsh.model.occ.addCylinder(x0 + (l0 + d0) / 2, y0, z0, -d0 / 2, 0.0, 0.0, d0 / 2)  # conn2
        gmsh.model.occ.addCylinder(x0 - l0 / 2, y0, z0, l0, 0.0, 0.0, d0 / 2)  # gauge

        # Synchronize
        gmsh.model.occ.synchronize()

        # Remove duplicate entities
        gmsh.model.occ.removeAllDuplicates()

        # Synchronize
        gmsh.model.occ.synchronize()

        # Get model entities
        points, _lines, _surfaces, _volumes = (gmsh.model.getEntities(d) for d in [0, 1, 2, 3])

        # Extract tags of relevant surfaces and volumes
        s0, s1, s2, s3 = 1, 5, 3, 7  # ordering by inspection
        v0, v1, v2, v3, v4 = 1, 3, 2, 4, 5  # ordering by inspection

        # Set mesh size via points
        gmsh.model.mesh.setSize(points, d0 / nr)  # heuristic

        # Generate the mesh
        gmsh.model.mesh.generate()
        # gmsh.model.mesh.optimize("Netgen")

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Define physical groups for subdomains (! target tag > 0)
        # domain = 1
        # gmsh.model.addPhysicalGroup(tdim, [v[1] for v in volumes], domain)
        # gmsh.model.setPhysicalName(tdim, domain, 'domain')
        domain_grip_left = 2
        gmsh.model.addPhysicalGroup(tdim, [v0, v1], domain_grip_left)
        gmsh.model.setPhysicalName(tdim, domain_grip_left, "domain_grip_left")
        domain_grip_right = 3
        gmsh.model.addPhysicalGroup(tdim, [v2, v3], domain_grip_right)
        gmsh.model.setPhysicalName(tdim, domain_grip_right, "domain_grip_right")
        domain_gauge = 4
        gmsh.model.addPhysicalGroup(tdim, [v4], domain_gauge)
        gmsh.model.setPhysicalName(tdim, domain_gauge, "domain_gauge")

        # Define physical groups for interfaces (! target tag > 0)
        # surface = 1
        # gmsh.model.addPhysicalGroup(tdim - 1, [s[1] for s in surfaces], surface)
        # gmsh.model.setPhysicalName(tdim - 1, surface, 'surface')
        surface_grip_left = 2
        gmsh.model.addPhysicalGroup(tdim - 1, [s0], surface_grip_left)
        gmsh.model.setPhysicalName(tdim - 1, surface_grip_left, "surface_grip_left")
        surface_grip_right = 3
        gmsh.model.addPhysicalGroup(tdim - 1, [s1], surface_grip_right)
        gmsh.model.setPhysicalName(tdim - 1, surface_grip_right, "surface_grip_right")
        surface_plane_left = 4
        gmsh.model.addPhysicalGroup(tdim - 1, [s2], surface_plane_left)
        gmsh.model.setPhysicalName(tdim - 1, surface_plane_left, "surface_plane_left")
        surface_plane_right = 5
        gmsh.model.addPhysicalGroup(tdim - 1, [s3], surface_plane_right)
        gmsh.model.setPhysicalName(tdim - 1, surface_plane_right, "surface_plane_right")

        gmsh.model.occ.synchronize()

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)
            # gmsh.write(name + ".step")

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    mesh_iso6892_gmshapi(msh_file="iso6892.msh")

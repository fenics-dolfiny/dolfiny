#!/usr/bin/env python3

from mpi4py import MPI


def mesh_tube3d_gmshapi(
    name="tube3d",
    r=1.0,
    t=0.2,
    h=1.0,
    nr=30,
    nt=6,
    nh=10,
    x0=0.0,
    y0=0.0,
    z0=0.0,
    do_quads=False,
    order=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 3d tube using the Python API of Gmsh.
    """
    tdim = 3  # target topological dimension

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.set_number("General.NumThreads", 1)  # reproducibility

        if do_quads:
            gmsh.option.set_number("Mesh.Algorithm", 8)
            gmsh.option.set_number("Mesh.Algorithm3D", 10)
            # gmsh.option.set_number("Mesh.SubdivisionAlgorithm", 2)
        else:
            gmsh.option.set_number("Mesh.Algorithm", 5)
            gmsh.option.set_number("Mesh.Algorithm3D", 4)
            gmsh.option.set_number("Mesh.AlgorithmSwitchOnFailure", 6)

        # Add model under given name
        gmsh.model.add(name)

        # Create points and line
        p0 = gmsh.model.occ.add_point(x0 + r, y0, 0.0)
        p1 = gmsh.model.occ.add_point(x0 + r + t, y0, 0.0)
        l0 = gmsh.model.occ.add_line(p0, p1)
        s0 = gmsh.model.occ.revolve(
            [(1, l0)],
            x0,
            y0,
            z0,
            0,
            0,
            1,
            angle=+gmsh.pi,
            numElements=[nr],
            recombine=do_quads,
        )
        s1 = gmsh.model.occ.revolve(
            [(1, l0)],
            x0,
            y0,
            z0,
            0,
            0,
            1,
            angle=-gmsh.pi,
            numElements=[nr],
            recombine=do_quads,
        )
        ring, _ = gmsh.model.occ.fuse([s0[1]], [s1[1]])
        tube = gmsh.model.occ.extrude(ring, 0, 0, h, [nh], recombine=do_quads)  # noqa: F841

        # Synchronize
        gmsh.model.occ.synchronize()

        # Get model entities
        points, lines, surfaces, volumes = (gmsh.model.occ.get_entities(d) for d in [0, 1, 2, 3])
        boundaries = gmsh.model.get_boundary(volumes, oriented=False)  # noqa: F841

        # Assertions, problem-specific
        assert len(volumes) == 2

        # Helper
        def extract_tags(a):
            return list(ai[1] for ai in a)

        # Extract certain tags, problem-specific
        tag_subdomains_total = extract_tags(volumes)

        # Set geometrical identifiers (obtained by inspection)
        tag_interfaces_lower = extract_tags([surfaces[0], surfaces[1]])
        tag_interfaces_upper = extract_tags([surfaces[6], surfaces[9]])
        tag_interfaces_inner = extract_tags([surfaces[2], surfaces[7]])
        tag_interfaces_outer = extract_tags([surfaces[3], surfaces[8]])

        # Define physical groups for subdomains (! target tag > 0)
        domain = 1
        gmsh.model.add_physical_group(tdim, tag_subdomains_total, domain)
        gmsh.model.set_physical_name(tdim, domain, "domain")

        # Define physical groups for interfaces (! target tag > 0)
        surface_lower = 1
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_lower, surface_lower)
        gmsh.model.set_physical_name(tdim - 1, surface_lower, "surface_lower")
        surface_upper = 2
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_upper, surface_upper)
        gmsh.model.set_physical_name(tdim - 1, surface_upper, "surface_upper")
        surface_inner = 3
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_inner, surface_inner)
        gmsh.model.set_physical_name(tdim - 1, surface_inner, "surface_inner")
        surface_outer = 4
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_outer, surface_outer)
        gmsh.model.set_physical_name(tdim - 1, surface_outer, "surface_outer")

        # Set refinement in radial direction
        gmsh.model.mesh.setTransfiniteCurve(l0, numNodes=nt)

        # Generate the mesh
        gmsh.model.mesh.generate()

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    mesh_tube3d_gmshapi(msh_file="tube3d.msh")

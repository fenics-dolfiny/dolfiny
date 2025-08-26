#!/usr/bin/env python3

from mpi4py import MPI

import dolfiny


def mesh_spanner_gmshapi(
    name="spanner",
    step_file="spanner25_lo.step",
    size=0.04,
    do_quads=False,
    order=2,
    msh_file=None,
    vtk_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 3d spanner from STEP file using the Python API of Gmsh.

    See Gmsh references:
    [1] https://gmsh.info/doc/texinfo/gmsh.html
    [2] https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/api/gmsh.py
    """
    tdim = 3  # target topological dimension

    q_measure, q_eps = "minDetJac", 0.0  # mesh quality metric
    # q_measure, q_eps = "minSJ", 0.1  # mesh quality metric

    success = True  # meshing succeeded with given quality metric

    # Perform Gmsh work only on rank = 0

    if comm.rank == 0:
        import gmsh

        # Initialise gmsh and set options
        gmsh.initialize()
        gmsh.option.set_number("General.Terminal", 1)
        gmsh.option.set_number("General.NumThreads", 1)  # reproducibility

        gmsh.option.set_string("Geometry.OCCTargetUnit", "M")
        gmsh.option.set_number("Geometry.OCCFixDegenerated", 1)
        gmsh.option.set_number("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.set_number("Geometry.OCCFixSmallFaces", 1)

        if do_quads:
            gmsh.option.set_number("Mesh.Algorithm", 8)
            gmsh.option.set_number("Mesh.Algorithm3D", 10)
            gmsh.option.set_number("Mesh.SubdivisionAlgorithm", 2)
        else:
            gmsh.option.set_number("Mesh.Algorithm", 5)
            gmsh.option.set_number("Mesh.Algorithm3D", 4)
            gmsh.option.set_number("Mesh.AlgorithmSwitchOnFailure", 6)

        if order < 2:
            print("WARNING: Check suitability of model for low-order meshing!")

            size *= 0.74
        else:
            pass

        # Perform mesh smoothing
        gmsh.option.set_number("Mesh.Smoothing", 3)

        # Add model under given name
        gmsh.model.add(name)

        # Create
        gmsh.model.occ.import_shapes(step_file)

        # Synchronize
        gmsh.model.occ.synchronize()

        # Get model entities
        points, lines, surfaces, volumes = (gmsh.model.occ.get_entities(d) for d in [0, 1, 2, 3])
        boundaries = gmsh.model.get_boundary(volumes, oriented=False)

        # Assertions, problem-specific
        assert len(volumes) == 1

        # Helper
        def extract_tags(a):
            return list(ai[1] for ai in a)

        # Extract certain tags, problem-specific
        tag_subdomains_total = extract_tags(volumes)

        # NOTE: STEP-inspected geometrical identifiers require shift by 1
        if step_file == "spanner25_lo.step":
            tag_interfaces_flats = extract_tags([surfaces[11 - 1], surfaces[14 - 1]])
            tag_interfaces_crown = extract_tags([surfaces[k - 1] for k in range(18, 42)])
            tag_interfaces_other = list(
                set(extract_tags(boundaries))
                - set(tag_interfaces_flats)
                - set(tag_interfaces_crown)
            )
        elif step_file == "spanner25_hi.step":
            tag_interfaces_flats = extract_tags([surfaces[4 - 1], surfaces[12 - 1]])
            tag_interfaces_crown = extract_tags([surfaces[k - 1] for k in range(97, 120 + 1)])
            tag_interfaces_other = list(
                set(extract_tags(boundaries))
                - set(tag_interfaces_flats)
                - set(tag_interfaces_crown)
            )
        else:
            raise RuntimeError(f"Cannot tag required entities for '{step_file}'")

        # Define physical groups for subdomains (! target tag > 0)
        domain = 1
        gmsh.model.add_physical_group(tdim, tag_subdomains_total, domain)
        gmsh.model.set_physical_name(tdim, domain, "domain")

        # Define physical groups for interfaces (! target tag > 0)
        surface_flats = 1
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_flats, surface_flats)
        gmsh.model.set_physical_name(tdim - 1, surface_flats, "surface_flats")
        surface_crown = 2
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_crown, surface_crown)
        gmsh.model.set_physical_name(tdim - 1, surface_crown, "surface_crown")
        surface_other = 3
        gmsh.model.add_physical_group(tdim - 1, tag_interfaces_other, surface_other)
        gmsh.model.set_physical_name(tdim - 1, surface_other, "surface_other")

        # Set sizes
        csize = size  # characteristic size (head diameter)
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.set_numbers(
            distance, "SurfacesList", tag_interfaces_flats + tag_interfaces_crown
        )
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.set_number(threshold, "InField", distance)
        gmsh.model.mesh.field.set_number(threshold, "SizeMin", csize * 0.05)
        gmsh.model.mesh.field.set_number(threshold, "SizeMax", csize * 0.20)
        gmsh.model.mesh.field.set_number(threshold, "DistMin", csize * 0.10)
        gmsh.model.mesh.field.set_number(threshold, "DistMax", csize * 0.30)
        gmsh.model.mesh.field.set_as_background_mesh(threshold)

        # Generate the mesh
        gmsh.model.mesh.generate()
        gmsh.model.mesh.optimize("Netgen")

        # Set geometric order of mesh cells
        gmsh.model.mesh.set_order(order)

        # Statistics and checks
        for d in [tdim - 1, tdim]:
            for e in gmsh.model.occ.get_entities(d):
                for t in gmsh.model.mesh.get_element_types(*e):
                    elements, _ = gmsh.model.mesh.get_elements_by_type(t, e[1])
                    element_name, _, _, _, _, _ = gmsh.model.mesh.get_element_properties(t)
                    elements_quality = gmsh.model.mesh.get_element_qualities(elements, q_measure)
                    below_eps = sum(elements_quality <= q_eps)

                    print(
                        f"{e!s:8s}: {len(elements):8d} {element_name:20s} ({t:2d}), "
                        + f"{q_measure:>8s} < {q_eps} = {below_eps:4d} "
                        + f"[{min(elements_quality):+.3e}, {max(elements_quality):+.3e}] "
                        + ("Quality warning!" if below_eps > 0 else ""),
                        flush=True,
                    )

                    # success &= not bool(below_eps)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)

        # Optional: Write vtk file
        if vtk_file is not None:
            gmsh.write(vtk_file)

    if not comm.bcast(success, root=0):
        exit()

    return gmsh.model if comm.rank == 0 else None, tdim


if __name__ == "__main__":
    msh_file = "spanner.msh"
    vtk_file = "spanner.vtk"

    mesh_spanner_gmshapi(msh_file=msh_file, vtk_file=vtk_file)

    import pyvista

    grid = pyvista.read(vtk_file)
    plotter = pyvista.Plotter(off_screen=True, theme=dolfiny.pyvista.theme)

    grid_surface_hires = grid.extract_surface(nonlinear_subdivision=4)
    plotter.add_mesh(grid_surface_hires, color="tab:orange")
    plotter.add_mesh(
        grid.separate_cells().extract_surface(nonlinear_subdivision=4).extract_feature_edges(),
        style="wireframe",
        color="black",
        line_width=dolfiny.pyvista.pixels // 1000,
        render_lines_as_tubes=True,
    )

    plotter.zoom_camera(1.3)
    plotter.show_axes()
    plotter.screenshot("spanner.png", transparent_background=False)

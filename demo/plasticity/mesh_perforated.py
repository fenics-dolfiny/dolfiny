from mpi4py import MPI


def mesh_perforated(name="perforated", comm=MPI.COMM_WORLD, clscale=0.2, extrude_z=0.1):
    if comm.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.model.add(name)

        holes = [
            ((0.18, 0.49), 0.030),
            ((0.28, 0.53), 0.026),
            ((0.38, 0.47), 0.028),
            ((0.50, 0.51), 0.032),
            ((0.62, 0.46), 0.027),
            ((0.72, 0.54), 0.029),
            ((0.82, 0.50), 0.025),
        ]

        plate = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)

        voids = []
        for (x_c, y_c), radius in holes:
            hole = gmsh.model.occ.addDisk(x_c, y_c, 0.0, radius, radius)
            voids.append((2, hole))

        cut_surfaces, _ = gmsh.model.occ.cut([(2, plate)], voids)
        extruded = gmsh.model.occ.extrude(cut_surfaces, 0, 0, extrude_z)

        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_min := clscale)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_max := 10 * clscale)

        gmsh.model.occ.synchronize()

        # Adaptive mesh sizing (with background mesh)
        strip_points = [
            gmsh.model.occ.addPoint(x_c, y_c, 0.5 * extrude_z) for (x_c, y_c), _ in holes
        ]

        gmsh.model.occ.synchronize()

        field_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_dist, "PointsList", strip_points)
        gmsh.model.mesh.field.setNumber(field_dist, "Sampling", 100)

        field_th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_th, "InField", field_dist)
        gmsh.model.mesh.field.setNumber(field_th, "SizeMin", h_min)
        gmsh.model.mesh.field.setNumber(field_th, "SizeMax", h_max)
        gmsh.model.mesh.field.setNumber(field_th, "DistMin", 0.00)
        gmsh.model.mesh.field.setNumber(field_th, "DistMax", 0.30)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_th)

        gmsh.model.addPhysicalGroup(3, [tag for dim, tag in extruded if dim == 3])

        gmsh.model.mesh.generate()
        gmsh.write(f"{name}.msh")

    return gmsh.model if comm.rank == 0 else None, 3

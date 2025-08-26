from mpi4py import MPI


def mesh_notched(name="notched", comm=MPI.COMM_WORLD, clscale=0.2):
    if comm.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.model.add(name)

        dx = 0.2
        dy = 0.05
        rect1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        rect2 = gmsh.model.occ.addRectangle(0, 0.5 - dy / 2, 0, dx, dy)

        gmsh.model.occ.cut([(2, rect1)], [(2, rect2)], tag=3)

        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [3])
        gmsh.option.setNumber("Mesh.MeshSizeFactor", clscale)
        gmsh.model.mesh.generate()
        gmsh.write("notched.msh")

    return gmsh.model if comm.rank == 0 else None, 2

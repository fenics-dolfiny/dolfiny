#!/usr/bin/env python3

import sys
from mpi4py import MPI
import gmsh
import numpy as np
import dolfinx

def mesh_Hertz3D_gmsh(
    cell_tags,
    facet_tags,
    nx=10,
    ny=10,
    nz=10,
    name="Hertz3D",
    R=1.0,
    H=1.0,
    L=2.0,
    W=2.0,
    H2=1.5,
    order=1,
    verbosity=1,
    msh_file=None,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 3d Hertz contact problem using the Python API of Gmsh.
    For details of the geometry, see https://doi.org/10.1016/j.jmps.2026.106617
    """
    tdim = 3  # topological dimension

    gmsh.initialize() # Initialize Gmsh instace

    # generate mesh only on rank = 0 
    if comm.rank == 0:
        
        # set options
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", verbosity)

        # Add model under given name
        gmsh.model.add(name)

        # Create geometry
        origin = (0.0, 0.0, 0.0)
        dx, dy, dz = W, L, H+H2
        
        # Box: half-space body (1/4 of the full domain)
        box_body = gmsh.model.occ.addBox(*origin, dx, dy, H)
        # Box: third medium (1/4 of the full domain)
        box_tm = gmsh.model.occ.addBox(0., 0., H, dx, dy, H2)
        # 1/4 of the sphere
        center = (0., 0., dz)
        sphere = gmsh.model.occ.addSphere(*center, R, angle1=-np.pi/2, angle2=0., angle3=np.pi/2)

        ## Box containing the sphere -- conformal meshing through boolean intersection
        whole_domain, map_to_input = gmsh.model.occ.fragment(
            [(tdim, box_body), (tdim, box_tm)], 
            [(tdim, sphere)]
            )
        # whole_domain, map_to_input = gmsh.model.occ.fragment([(tdim, box_tm)], [(tdim, sphere)])
        gmsh.model.occ.synchronize()


        # Helper
        def extract_tags(a):
            return list(ai[1] for ai in a)

        def select_boundary_face_by_z(volume_tag, z_target, tol=1.0e-6):
            faces = gmsh.model.getBoundary([(tdim, volume_tag)], combined=False, oriented=False)
            for (dim, face_tag) in faces:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, face_tag)
                if abs(zmin - z_target) < tol and abs(zmax - z_target) < tol:
                    return face_tag
            raise RuntimeError(f"Could not find boundary face at z={z_target} for volume {volume_tag}")

        # Get model entities
        lines, surfaces, volumes = (gmsh.model.getEntities(d) for d in [1, 2, 3])

        # for vol in volumes:
        #     bnd = gmsh.model.getBoundary([vol], oriented=False)
        #     print(vol, len(bnd))

        # Assertion to test correct number of entities
        assert len(volumes) == 3

        # if '-nopopup' not in sys.argv:
        #     gmsh.fltk.run() 

        # Extract the volume tags for the physical groups
        body_vol = [idx for (dim, idx) in map_to_input[0]][0]    # fragment of box_body
        medium_vol = [idx for (dim, idx) in map_to_input[1]][0]  # fragment of box_tm
        sphere_vol = [idx for (dim, idx) in map_to_input[2]][0]  # fragment of sphere
    
        # Define physical groups for subdomains
        gmsh.model.addPhysicalGroup(tdim, [body_vol], tag=cell_tags["body"], name="body")
        gmsh.model.addPhysicalGroup(tdim, [medium_vol], tag=cell_tags["tm"], name="third_medium")
        gmsh.model.addPhysicalGroup(tdim, [sphere_vol], tag=cell_tags["indenter"], name="indenter")

        ## Extract surface tags (obtained by visual inspection of the geometry in Gmsh -- NOT ROBUST)
        #sphere_top = select_boundary_face_by_z(sphere_vol, dz)
        sphere_top = extract_tags([surfaces[12]])  # top surface of the sphere
        bottom = extract_tags([surfaces[11]])  # bottom surface of the box
        x_symm = extract_tags([surfaces[0], surfaces[7], surfaces[14]])  # x-symmetry plane
        y_symm = extract_tags([surfaces[1], surfaces[9], surfaces[13]])  # y-symmetry plane
        body_top = extract_tags([surfaces[5]])  # top surface of the box (interface with third medium)
    
        # Define physical groups for surfaces
        gmsh.model.addPhysicalGroup(tdim - 1, sphere_top, tag=facet_tags["sphere_top"], name="sphere_top")
        gmsh.model.addPhysicalGroup(tdim - 1, bottom, tag=facet_tags["bottom"], name="bottom")
        gmsh.model.addPhysicalGroup(tdim - 1, x_symm, tag=facet_tags["x_symm"], name="x_symmetry")
        gmsh.model.addPhysicalGroup(tdim - 1, y_symm, tag=facet_tags["y_symm"], name="y_symmetry")
        gmsh.model.addPhysicalGroup(tdim - 1, body_top, tag=facet_tags["body_top"], name="body_top")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)
        
        # # Generate mesh
        gmsh.model.mesh.generate()

        # # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(order)

        # Optional: Write msh file
        if msh_file is not None:
            gmsh.write(msh_file)
        
    return gmsh.model if comm.rank == 0 else None, tdim
        


# if __name__ == "__main__":
    
#     cell_tags = {"body": 1, "tm": 2, "indenter": 3, "sphere_top": 4}
#     facet_tags = {"sphere_top": 10, "bottom": 11, "x_symm": 12, "y_symm": 13, "body_top": 14}
#     mesh_Hertz3D_gmsh(cell_tags, facet_tags)
# #     mesh_data = mesh_Hertz3D_gmsh(cell_tags, facet_tags)
#     mesh = mesh_data.mesh
#     cell_markers = mesh_data.cell_tags
#     facet_markers = mesh_data.facet_tags

#     # # write mesh to Paraview for visualization
#     with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "Hertz3D.xdmf", "w") as xdmf:
#         xdmf.write_mesh(mesh)
#         xdmf.write_meshtags(cell_markers, mesh.geometry)
#         xdmf.write_meshtags(facet_markers, mesh.geometry)


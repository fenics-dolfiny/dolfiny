#!/usr/bin/env python3

import sys
import os
from mpi4py import MPI
import gmsh
import numpy as np
import dolfinx

def mesh_double_arch_gmsh(
    cell_tags,
    facet_tags,
    tdim=3,  # topological dimension (2 if 2D plane-strain, 3 if 3D)
    Lx=260,
    Lz=50,
    H=50,
    Di=90,
    lR=5,
    t=5,
    g0=20,
    nL=52, # number of elements along L
    nH=10, # number of elements along H
    nt=1, # number of elements along t
    nz=25, # number of elements along z
    nDi=24, # number of elements along Di
    nTM=10, # number of elements along third medium (i.e. from block to arches)
    name="double_arch",
    vtk_file=False,
    order=1,
    verbosity=1,
    comm=MPI.COMM_WORLD,
):
    """
    Create mesh of 3d double arch problem using the Python API of Gmsh.
    For details of the geometry, see http://dx.doi.org/10.1016/j.compstruc.2015.02.027
    """

    gmsh.initialize() # Initialize Gmsh instace

    center = (0., g0+Di/2+2*t, 0.)
    R_int = Di/2
    R_ext = Di/2 + 2*t
    R_mid = Di/2 + t

    x3, y3 = center[0]-R_int*np.cos(np.pi/4), center[1]-R_int*np.sin(np.pi/4)
    x4, y4 = center[0]+R_int*np.cos(np.pi/4), center[1]-R_int*np.sin(np.pi/4)

    x5, y5 = center[0]-R_ext*np.cos(np.pi/4), center[1]-R_ext*np.sin(np.pi/4)
    x6, y6 = center[0]+R_ext*np.cos(np.pi/4), center[1]-R_ext*np.sin(np.pi/4)

    x7, y7 = center[0]-R_mid*np.cos(np.pi/4), center[1]-R_mid*np.sin(np.pi/4)
    x8, y8 = center[0]+R_mid*np.cos(np.pi/4), center[1]-R_mid*np.sin(np.pi/4)

    # generate mesh only on rank = 0
    if comm.rank == 0:
        # set options
        gmsh.option.setNumber("General.Verbosity", verbosity)

        # Add model under given name
        gmsh.model.add(name)

        # create 2D geometry (block + double arch + third-medium) -- to be extruded in z-direction
        # create block from points and lines
        pb_1 = gmsh.model.occ.addPoint(-Lx/2, -H, 0.)
        pb_2 = gmsh.model.occ.addPoint(Lx/2, -H, 0.)

        # create double arch from points
        # first (internal) arch
        center_point = gmsh.model.occ.addPoint(*center)
        x3_p = gmsh.model.occ.addPoint(x3, y3, 0.)
        x4_p = gmsh.model.occ.addPoint(x4, y4, 0.)
        p1_left = gmsh.model.occ.addPoint(-Di/2, center[1], 0.)
        p1_middle = gmsh.model.occ.addPoint(0., center[1]-(Di/2), 0.)
        p1_right = gmsh.model.occ.addPoint(Di/2, center[1], 0.)
        # arch1 = gmsh.model.occ.addCircleArc(p1_left, p1_middle, p1_right, center=False)
        arch1_l = gmsh.model.occ.addCircleArc(p1_left, center_point, x3_p)
        arch1_m = gmsh.model.occ.addCircleArc(x3_p, center_point, x4_p)
        arch1_r = gmsh.model.occ.addCircleArc(x4_p, center_point, p1_right)
        arch1 = (arch1_l, arch1_m, arch1_r)

        # external arch
        x5_p = gmsh.model.occ.addPoint(x5, y5, 0.)
        x6_p = gmsh.model.occ.addPoint(x6, y6, 0.)
        p2_left = gmsh.model.occ.addPoint(-Di/2-2*t, center[1], 0.)
        p2_middle = gmsh.model.occ.addPoint(0., center[1]-(Di/2+2*t), 0.)
        p2_right = gmsh.model.occ.addPoint(Di/2+2*t, center[1], 0.)
        # arch2 = gmsh.model.occ.addCircleArc(p2_left, p2_middle, p2_right, center=False)
        arch2_l = gmsh.model.occ.addCircleArc(p2_left, center_point, x5_p)
        arch2_m = gmsh.model.occ.addCircleArc(x5_p, center_point, x6_p)
        arch2_r = gmsh.model.occ.addCircleArc(x6_p, center_point, p2_right)
        arch2_plus = (arch2_l, arch2_m, arch2_r)
        arch2_min = (-arch2_r, -arch2_m, -arch2_l)

        # middle arch
        xm_l = gmsh.model.occ.addPoint(x7, y7, 0.)
        xm_r = gmsh.model.occ.addPoint(x8, y8, 0.)
        pm_left = gmsh.model.occ.addPoint(-Di/2-t, center[1], 0.)
        pm_middle = gmsh.model.occ.addPoint(0., center[1]-(Di/2+t), 0.)
        pm_right = gmsh.model.occ.addPoint(Di/2+t, center[1], 0.)
        # archm = gmsh.model.occ.addCircleArc(pm_left, pm_middle, pm_right, center=False)
        archm_l = gmsh.model.occ.addCircleArc(pm_left, center_point, xm_l)
        archm_m = gmsh.model.occ.addCircleArc(xm_l, center_point, xm_r)
        archm_r = gmsh.model.occ.addCircleArc(xm_r, center_point, pm_right)
        archm_plus = (archm_l, archm_m, archm_r)
        archm_min = (-archm_r, -archm_m, -archm_l)

        # third medium
        # external (between arches and block)
        ptm_1 = gmsh.model.occ.addPoint(-Lx/2, center[1], 0.)
        ptm_2 = gmsh.model.occ.addPoint(-Lx/2, 0., 0.)
        origin = gmsh.model.occ.addPoint(0., 0., 0.)
        ptm_3 = gmsh.model.occ.addPoint(Lx/2, 0., 0.)
        ptm_4 = gmsh.model.occ.addPoint(Lx/2, center[1], 0.)
        # internal (above arches)
        # rectangle for structured mesh
        ptm_r1 = gmsh.model.occ.addPoint(-lR, center[1], 0.)
        ptm_r2 = gmsh.model.occ.addPoint(-lR, center[1]-2*lR, 0.)
        ptm_r3 = gmsh.model.occ.addPoint(lR, center[1]-2*lR, 0.)
        ptm_r4 = gmsh.model.occ.addPoint(lR, center[1], 0.)

        # add curve loops and surfaces for each subdomain
        # DOUBLE ARCH
        l1_t_r = gmsh.model.occ.addLine(p1_right, pm_right)
        l1_t_l = gmsh.model.occ.addLine(pm_left, p1_left)
        a1_cl = gmsh.model.occ.addCurveLoop([*arch1, l1_t_r, *archm_min, l1_t_l])
        a1_surf = gmsh.model.occ.addPlaneSurface([a1_cl])

        l2_t_r = gmsh.model.occ.addLine(pm_right, p2_right)
        l2_t_l = gmsh.model.occ.addLine(p2_left, pm_left)
        a2_cl = gmsh.model.occ.addCurveLoop([*archm_plus, l2_t_r, *arch2_min, l2_t_l])
        a2_surf = gmsh.model.occ.addPlaneSurface([a2_cl])

        # THIRD-MEDIUM
        # EXTERNAL portion
        # left portion
        tm_l1 = gmsh.model.occ.addLine(ptm_1, p2_left)
        tm_l2 = arch2_l
        tm_l3 = gmsh.model.occ.addLine(x5_p, ptm_2)
        tm_l4 = gmsh.model.occ.addLine(ptm_2, ptm_1)
        tm_cl = gmsh.model.occ.addCurveLoop([tm_l1, tm_l2, tm_l3, tm_l4])
        tm_surf_L = gmsh.model.occ.addPlaneSurface([tm_cl])

        # central portion
        tm_c1 = arch2_m
        tm_c2 = gmsh.model.occ.addLine(x6_p, ptm_3)
        tm_c3 = gmsh.model.occ.addLine(ptm_3, ptm_2)
        tm_cl_c = gmsh.model.occ.addCurveLoop([-tm_l3, tm_c1, tm_c2, tm_c3])
        tm_surf_C = gmsh.model.occ.addPlaneSurface([tm_cl_c])

        # right portion
        tm_r1 = arch2_r
        tm_r2 = gmsh.model.occ.addLine(p2_right, ptm_4)
        tm_r3 = gmsh.model.occ.addLine(ptm_4, ptm_3)
        tm_cl_r = gmsh.model.occ.addCurveLoop([tm_r1, tm_r2, tm_r3, -tm_c2])
        tm_surf_R = gmsh.model.occ.addPlaneSurface([tm_cl_r])

        # INTERNAL portion
        # sector 1
        tm_i1 = gmsh.model.occ.addLine(p1_left, ptm_r1)
        tm_i2 = gmsh.model.occ.addLine(ptm_r1, ptm_r2)
        tm_i3 = gmsh.model.occ.addLine(ptm_r2, x3_p)
        tm_i4 = - arch1_l
        tm_cl_i1 = gmsh.model.occ.addCurveLoop([tm_i1, tm_i2, tm_i3, tm_i4])
        tm_surf_s1 = gmsh.model.occ.addPlaneSurface([tm_cl_i1])

        # sector 2
        tm_i5 = gmsh.model.occ.addLine(ptm_r2, ptm_r3)
        tm_i6 = gmsh.model.occ.addLine(ptm_r3, x4_p)
        tm_i7 = - arch1_m
        tm_cl_i2 = gmsh.model.occ.addCurveLoop([tm_i5, tm_i6, tm_i7, -tm_i3])
        tm_surf_s2 = gmsh.model.occ.addPlaneSurface([tm_cl_i2])

        # sector 3
        tm_i8 = gmsh.model.occ.addLine(ptm_r3, ptm_r4)
        tm_i9 = gmsh.model.occ.addLine(ptm_r4, p1_right)
        tm_i10 = - arch1_r
        tm_cl_i3 = gmsh.model.occ.addCurveLoop([tm_i8, tm_i9, tm_i10, -tm_i6])
        tm_surf_s3 = gmsh.model.occ.addPlaneSurface([tm_cl_i3])

        # sector - rectangle
        tm_r4_line = gmsh.model.occ.addLine(ptm_r1, ptm_r4)
        tm_cl_r = gmsh.model.occ.addCurveLoop([-tm_i2, tm_r4_line, -tm_i8, -tm_i5])
        tm_surf_r = gmsh.model.occ.addPlaneSurface([tm_cl_r])

        # BLOCK
        b_l1 = gmsh.model.occ.addLine(pb_1, ptm_2)
        b_l2 = -tm_c3
        b_l4 = gmsh.model.occ.addLine(ptm_3, pb_2)
        b_l5 = gmsh.model.occ.addLine(pb_2, pb_1)
        b_cl = gmsh.model.occ.addCurveLoop([b_l1, b_l2, b_l4, b_l5])
        b_surf = gmsh.model.occ.addPlaneSurface([b_cl])

        # synchronize the CAD kernel with the Gmsh model
        gmsh.model.occ.synchronize()


        ## structured mesh
        N_h = nH+1
        N_L = nL+1
        N_t = nt+1
        N_Di = nDi+1
        N_TM = nTM+1

        ## block
        gmsh.model.mesh.setTransfiniteCurve(b_l1, N_h)
        gmsh.model.mesh.setTransfiniteCurve(b_l2, N_L)
        gmsh.model.mesh.setTransfiniteCurve(b_l4, N_h)
        gmsh.model.mesh.setTransfiniteCurve(b_l5, N_L)

        # double arch
        gmsh.model.mesh.setTransfiniteCurve(tm_c1, N_L)
        gmsh.model.mesh.setTransfiniteCurve(tm_i7, N_L)
        gmsh.model.mesh.setTransfiniteCurve(archm_m, N_L)
        gmsh.model.mesh.setTransfiniteCurve(arch1_m, N_L)
        gmsh.model.mesh.setTransfiniteCurve(arch2_m, N_L)


        gmsh.model.mesh.setTransfiniteCurve(tm_l2, N_Di)
        gmsh.model.mesh.setTransfiniteCurve(tm_r1, N_Di)
        gmsh.model.mesh.setTransfiniteCurve(tm_i4, N_Di)
        gmsh.model.mesh.setTransfiniteCurve(tm_i10, N_Di)
        gmsh.model.mesh.setTransfiniteCurve(archm_l, N_Di)
        gmsh.model.mesh.setTransfiniteCurve(archm_r, N_Di)
        gmsh.model.mesh.setTransfiniteCurve(arch1_l, N_Di) 
        gmsh.model.mesh.setTransfiniteCurve(arch1_r, N_Di)
        gmsh.model.mesh.setTransfiniteCurve(arch2_l, N_Di) 
        gmsh.model.mesh.setTransfiniteCurve(arch2_r, N_Di) 


        gmsh.model.mesh.setTransfiniteCurve(l1_t_l, N_t)
        gmsh.model.mesh.setTransfiniteCurve(l1_t_r, N_t)
        gmsh.model.mesh.setTransfiniteCurve(l2_t_l, N_t)
        gmsh.model.mesh.setTransfiniteCurve(l2_t_r, N_t)

        # third medium
        # external part
        gmsh.model.mesh.setTransfiniteCurve(tm_l1, N_TM)
        gmsh.model.mesh.setTransfiniteCurve(tm_l3, N_TM)
        gmsh.model.mesh.setTransfiniteCurve(tm_l4, N_Di)

        gmsh.model.mesh.setTransfiniteCurve(tm_r1, N_Di)
        gmsh.model.mesh.setTransfiniteCurve(tm_r2, N_TM)
        gmsh.model.mesh.setTransfiniteCurve(tm_r3, N_Di)
        gmsh.model.mesh.setTransfiniteCurve(tm_c2, N_TM)
        # internal part
        gmsh.model.mesh.setTransfiniteCurve(tm_i1, N_TM)
        gmsh.model.mesh.setTransfiniteCurve(tm_i3, N_TM)
        gmsh.model.mesh.setTransfiniteCurve(tm_i6, N_TM)
        gmsh.model.mesh.setTransfiniteCurve(tm_i9, N_TM) 

        gmsh.model.mesh.setTransfiniteCurve(tm_i2, N_Di)   
        gmsh.model.mesh.setTransfiniteCurve(tm_i8, N_Di)

        gmsh.model.mesh.setTransfiniteCurve(tm_i5, N_L)
        gmsh.model.mesh.setTransfiniteCurve(tm_r4_line, N_L)


        ###
        surfaces = [b_surf, tm_surf_L, tm_surf_C, tm_surf_R, tm_surf_s1, tm_surf_s2, tm_surf_s3, tm_surf_r]
        for s in surfaces:
            gmsh.model.mesh.setTransfiniteSurface(s)
            gmsh.model.mesh.setRecombine(2, s)  # recombine triangles into quads

        gmsh.model.mesh.setTransfiniteSurface(a1_surf, cornerTags=[p1_left, p1_right, pm_right, pm_left])
        gmsh.model.mesh.setTransfiniteSurface(a2_surf, cornerTags=[pm_left, pm_right, p2_right, p2_left])
        gmsh.model.mesh.setRecombine(2, a1_surf)  
        gmsh.model.mesh.setRecombine(2, a2_surf)  
        

    eps = 1e-6  # tolerance for bounding box

    if tdim == 3:
        surfaces = [a1_surf, a2_surf, b_surf, tm_surf_L, tm_surf_C, tm_surf_R,
                    tm_surf_s1, tm_surf_s2, tm_surf_s3, tm_surf_r]
        surface_entities = [(2, s) for s in surfaces]

        volumes_entities = gmsh.model.occ.extrude(
            surface_entities, 0, 0, Lz, numElements=[nz], recombine=True
        )
        gmsh.model.occ.synchronize()

        vols = [tag for dim, tag in volumes_entities if dim == 3]
        vol_arch1, vol_arch2 = vols[0], vols[1]
        vol_block = vols[2]
        vol_tm = vols[3:]

        cell_entities = {
            "block": [vol_block],
            "arch1": [vol_arch1],
            "arch2": [vol_arch2],
            "tm":    vol_tm,
        }
        cell_dim, facet_dim = 3, 2
        z_min, z_max = -eps, Lz + eps

    else:  # tdim == 2, plane-strain: no extrusion, tag the 2D section directly
        gmsh.model.occ.synchronize()

        cell_entities = {
            "block": [b_surf],
            "arch1": [a1_surf],
            "arch2": [a2_surf],
            "tm":    [tm_surf_L, tm_surf_C, tm_surf_R,
                    tm_surf_s1, tm_surf_s2, tm_surf_s3, tm_surf_r],
        }
        cell_dim, facet_dim = 2, 1
        z_min, z_max = -eps, eps  # geometry lives at z = 0 only

    # --- cell (region) physical groups: identical call for both cases ---
    for key, ents in cell_entities.items():
        gmsh.model.addPhysicalGroup(cell_dim, ents, cell_tags[key], name=key)

    # --- facet (boundary) physical groups: same bounding-box logic, dim/z parametrized ---
    bottom_block = gmsh.model.getEntitiesInBoundingBox(
        -Lx/2 - eps, -H - eps, z_min,
        Lx/2 + eps, -H + eps, z_max,
        dim=facet_dim
    )
    gmsh.model.addPhysicalGroup(facet_dim, [t for _, t in bottom_block],
                                facet_tags["bottom"], name="bottom_block")

    top_arches_surf_L = gmsh.model.getEntitiesInBoundingBox(
        -Di/2 - 2*t - eps, center[1] - eps, z_min,
        -Di/2 + 2*t + eps, center[1] + eps, z_max,
        dim=facet_dim
    )
    top_arches_surf_R = gmsh.model.getEntitiesInBoundingBox(
        Di/2 - 2*t - eps, center[1] - eps, z_min,
        Di/2 + 2*t + eps, center[1] + eps, z_max,
        dim=facet_dim
    )
    gmsh.model.addPhysicalGroup(
        facet_dim,
        [t for _, t in top_arches_surf_L + top_arches_surf_R],
        facet_tags["top_arches"], name="top_arches"
    )

    # generate mesh (tdim already picks 2D vs 3D correctly)
    gmsh.model.mesh.generate(tdim)
    gmsh.model.mesh.setOrder(order)

    if vtk_file:
        if not os.path.exists(name):
                os.makedirs(name)
            
        gmsh.write(f"{name}/{name}.vtk")
        print(f"Mesh saved to {name}/{name}.vtk")

    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()

    return gmsh.model if comm.rank == 0 else None


# if __name__ == "__main__":
#     cell_tags = {"block": 1, "arch1": 2, "arch2": 3, "tm": 4}
#     facet_tags = {"bottom": 1, "top_arches": 2}
#     mesh_double_arch_gmsh(cell_tags=cell_tags, facet_tags=facet_tags, nt=2)


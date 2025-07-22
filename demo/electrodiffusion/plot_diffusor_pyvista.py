#!/usr/bin/env python3

from mpi4py import MPI

import pyvista


class Xdmf3Reader(pyvista.XdmfReader):
    _vtk_module_name = "vtkIOXdmf3"
    _vtk_class_name = "vtkXdmf3Reader"


def plot_diffusor_pyvista(name, xdmf_file=None, plot_file=None, options={}, comm=MPI.COMM_WORLD):
    if comm.rank > 0:
        return

    if xdmf_file is None:
        xdmf_file = f"./{name}.xdmf"  # NOTE: pre-pended "./"

    if plot_file is None:
        plot_file = f"./{name}.png"  # default: png

    # Read results and plot using pyvista (all in serial, on rank = 0)

    reader = Xdmf3Reader(path=xdmf_file)
    multiblock = reader.read()

    grid = multiblock[-1]
    grid.point_data["c"] = multiblock[0].point_data["c/0"]
    grid.point_data["φ"] = multiblock[1].point_data["φ/0"]

    pixels = 2048
    plotter = pyvista.Plotter(off_screen=True, window_size=[pixels, pixels], image_scale=1)
    plotter.add_axes(labels_off=True)

    sargs = dict(
        height=0.05,
        width=0.8,
        position_x=0.1,
        position_y=0.90,
        title="electrostatic potential [V]",
        font_family="courier",
        fmt="%1.2f",
        color="black",
        title_font_size=pixels // 50,
        label_font_size=pixels // 50,
    )

    if not grid.get_cell(0).is_linear:
        levels = 4
    else:
        levels = 0

    s = plotter.add_mesh(
        grid.extract_surface(nonlinear_subdivision=levels),
        scalars="φ",
        scalar_bar_args=sargs,
        cmap="coolwarm",
        specular=0.5,
        specular_power=20,
        smooth_shading=True,
        split_sharp_edges=True,
    )

    s.mapper.scalar_range = [0.0, 0.14]

    plotter.add_mesh(
        grid.separate_cells().extract_surface(nonlinear_subdivision=levels).extract_feature_edges(),
        style="wireframe",
        color="black",
        line_width=pixels // 1000,
        render_lines_as_tubes=True,
    )

    shift = 0.0025
    plotter.camera.focal_point = (-shift, shift, 0.0)
    plotter.camera.position = (-1.5 - shift, 1.0 + shift, -1.0 + 0.0)
    plotter.camera.up = (-1.0, 0.0, 0.0)
    plotter.zoom_camera(180)

    plotter.screenshot(plot_file, transparent_background=False)


if __name__ == "__main__":
    plot_diffusor_pyvista(name="diffusor_4species_steadystate")

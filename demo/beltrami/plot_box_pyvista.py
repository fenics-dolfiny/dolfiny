#!/usr/bin/env python3

from mpi4py import MPI

import pyvista


class Xdmf3Reader(pyvista.XdmfReader):
    _vtk_module_name = "vtkIOXdmf3"
    _vtk_class_name = "vtkXdmf3Reader"


def plot_box_pyvista(name, xdmf_file=None, plot_file=None, options={}, comm=MPI.COMM_WORLD):
    if comm.rank > 0:
        return

    if xdmf_file is None:
        xdmf_file = f"./{name}.xdmf"  # NOTE: pre-pended "./"

    if plot_file is None:
        plot_file = f"./{name}.png"  # default: png

    options_default = dict(
        on_deformed=True,
        u_factor=500.0,
        s_range=None,
    )
    options = options_default | options

    # Read results and plot using pyvista (all in serial, on rank = 0)

    reader = Xdmf3Reader(path=xdmf_file)
    multiblock = reader.read()

    grid = multiblock[-1]

    grid.point_data["u"] = multiblock[0].point_data["u/0"]
    grid.point_data["S"] = multiblock[1].point_data["S/0"]
    grid.point_data["s"] = multiblock[2].point_data["s/0"]

    pixels = 2048
    plotter = pyvista.Plotter(off_screen=True, window_size=[pixels, pixels], image_scale=1)
    plotter.add_axes(labels_off=True)

    sargs = dict(
        height=0.05,
        width=0.8,
        position_x=0.1,
        position_y=0.90,
        title=options["s_title"],
        font_family="courier",
        fmt="%1.2f",
        color="black",
        title_font_size=pixels // 50,
        label_font_size=pixels // 50,
    )

    factor = options["u_factor"]  # scaling factor, warped deformation

    plotter.add_text(
        f"u × {factor:.1f}",
        position=(pixels // 50, pixels // 50),
        font_size=pixels // 100,
        color="black",
        font="courier",
    )

    if options["on_deformed"]:
        grid_warped = grid.warp_by_vector("u", factor=factor)
    else:
        grid_warped = grid

    if not grid.get_cell(0).is_linear:
        levels = 4
    else:
        levels = 0

    s = plotter.add_mesh(
        grid_warped.extract_surface(nonlinear_subdivision=levels),
        scalars="s",
        scalar_bar_args=sargs,
        cmap="coolwarm",
        specular=0.5,
        specular_power=20,
        smooth_shading=True,
        split_sharp_edges=True,
    )

    if options["s_range"]:
        s.mapper.scalar_range = options["s_range"]

    plotter.add_mesh(
        grid_warped.separate_cells()
        .extract_surface(nonlinear_subdivision=levels)
        .extract_feature_edges(),
        style="wireframe",
        color="black",
        line_width=pixels // 1000,
        render_lines_as_tubes=True,
    )

    plotter.zoom_camera(1.0)

    plotter.screenshot(plot_file, transparent_background=False)


if __name__ == "__main__":
    plot_box_pyvista(name="solid_stressonly")

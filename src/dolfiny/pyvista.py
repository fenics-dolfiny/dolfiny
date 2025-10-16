from pyvista import themes

theme = themes.DocumentTheme()

pixels = 4 * 1024
theme.window_size = (pixels, pixels)

theme.axes.show = True
theme.camera.parallel_projection = True
theme.cmap = "coolwarm"
theme.jupyter_backend = "static"
theme.lighting_params.specular = 0.2
theme.lighting_params.specular_power = 10
theme.split_sharp_edges = True

theme.font.family = "courier"
theme.font.fmt = "%1.2f"
theme.font.label_size = pixels // 50
theme.font.size = pixels // 50
theme.font.title_size = pixels // 50

theme.colorbar_horizontal.height = 0.05
theme.colorbar_horizontal.position_x = 0.1
theme.colorbar_horizontal.position_y = 0.9
theme.colorbar_horizontal.width = 0.8
theme.colorbar_orientation = "horizontal"

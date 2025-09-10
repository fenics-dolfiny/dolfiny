class Plotter:
    def __init__(self, outfile, title=r"beam"):
        self.outfile = outfile

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        ax1.set_title(title, fontsize=12)
        ax1.set_xlabel(r"coordinate $x$, displacement $[m]$", fontsize=12)
        ax1.set_ylabel(r"coordinate $z$, displacement $[m]$", fontsize=12)
        ax1.invert_yaxis()
        ax1.axis("equal")
        ax1.grid(linewidth=0.25)

        fig.tight_layout()

        self.fig = fig
        self.ax1 = ax1

    def add(self, mesh, q, m, μ):
        x0, (ui, wi, _ri) = self.interpolate_on_mesh(mesh, q, m)

        color = (0.5 + μ.value * 0.5, 0.5 - μ.value * 0.5, 0.5 - μ.value * 0.5)
        endco = (0.5 - μ.value * 0.5, 0.5 - μ.value * 0.5, 0.5 + μ.value * 0.5)
        label = "undeformed" if μ.value == 0.0 else "deformed" if μ.value == 1.0 else None

        # Plot line
        self.ax1.plot(x0[:, 0] + ui, x0[:, 2] + wi, "-", linewidth=0.75, color=color, label=label)
        # Plot outer element nodes
        self.ax1.plot(
            x0[:: q + 1, 0] + ui[:: q + 1],
            x0[:: q + 1, 2] + wi[:: q + 1],
            ".",
            markersize=2.5,
            color=color,
        )
        # Plot marker at end
        self.ax1.plot(x0[-1, 0] + ui[-1], x0[-1, 2] + wi[-1], "o", markersize=3.0, color=endco)

        self.ax1.legend(loc="upper left")

        self.fig.savefig(self.outfile)

    def interpolate_on_mesh(self, mesh, q, u):
        # Extract mesh geometry nodal coordinates
        dm = mesh.geometry.dofmap
        oq = [0, *range(2, q + 1), 1]  # reorder lineX nodes: all ducks in a row...
        x0_idx = dm[:, oq].flatten()
        x0 = mesh.geometry.x[x0_idx]

        # Interpolate solution at mesh geometry nodes
        import dolfinx

        import dolfiny

        Q = dolfinx.fem.functionspace(mesh, ("P", q))
        uf = dolfinx.fem.Function(Q)

        if isinstance(u, list):
            ui = []
            for u_ in u:
                dolfiny.interpolation.interpolate(u_, uf)
                ui.append(uf.x.array[x0_idx])
        else:
            dolfiny.interpolation.interpolate(u, uf)
            ui = uf.x.array[x0_idx]

        return x0, ui

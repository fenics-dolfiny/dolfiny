import logging

import dolfinx
import dolfinx.io.gmsh


class XDMFFile(dolfinx.io.XDMFFile):
    KEYS_OF_MESHTAGS = "KeysOfMeshTags"

    def write_mesh_meshtags(self, mesh, mts=None):
        """Write mesh and meshtags to XDMFFile.

        Parameters
        ----------
        mesh:
            The dolfin mesh
        mts: optional
            The dict of MeshTags

        """
        logger = logging.getLogger("dolfiny")

        logger.debug("Writing mesh")
        self.write_mesh(mesh)

        if mts is None:
            return

        keys_meshtags = {key: mt.dim for key, mt in mts.items()}

        logger.debug("Writing information")
        self.write_information(self.KEYS_OF_MESHTAGS, str(keys_meshtags))

        logger.debug("Writing meshtags")
        for mt in mts.values():
            mesh.topology.create_connectivity(mt.dim, mesh.topology.dim)
            self.write_meshtags(mt, mesh.geometry)

    def write_mesh_data(self, mesh_data: dolfinx.io.gmsh.MeshData) -> None:
        """Write mesh with meshtags to XDMFFile.

        Parameters
        ----------
        mesh_data:
            A dolfinx MeshData object.

        """
        logger = logging.getLogger("dolfiny")

        logger.debug("Writing mesh")
        mesh = mesh_data.mesh
        self.write_mesh(mesh)

        logger.debug("Writing information")
        pg = mesh_data.physical_groups
        self.write_information(self.KEYS_OF_MESHTAGS, str(pg.keys()))

        logger.debug("Writing cell meshtags")
        # TODO: former connectivity computation?
        if (ct := mesh_data.cell_tags) is not None:
            self.write_meshtags(ct, mesh.geometry)

        if (ft := mesh_data.facet_tags) is not None:
            self.write_meshtags(ft, mesh.geometry)

        if (rt := mesh_data.ridge_tags) is not None:
            self.write_meshtags(rt, mesh.geometry)

        if (pt := mesh_data.peak_tags) is not None:
            self.write_meshtags(pt, mesh.geometry)

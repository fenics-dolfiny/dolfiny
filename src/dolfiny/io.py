import logging

import dolfinx
import dolfinx.io.gmshio


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

        logger.info("Writing mesh")
        self.write_mesh(mesh)

        if mts is None:
            return

        keys_meshtags = {key: mt.dim for key, mt in mts.items()}

        logger.info("Writing information")
        self.write_information(self.KEYS_OF_MESHTAGS, str(keys_meshtags))

        logger.info("Writing meshtags")
        for mt in mts.values():
            mesh.topology.create_connectivity(mt.dim, mesh.topology.dim)
            self.write_meshtags(mt, mesh.geometry)

    def write_mesh_data(self, mesh_data: dolfinx.io.gmshio.MeshData) -> None:
        """Write mesh with meshtags to XDMFFile.

        Parameters
        ----------
        mesh_data:
            A dolfinx MeshData object.

        """
        logger = logging.getLogger("dolfiny")

        logger.info("Writing mesh")
        mesh = mesh_data.mesh
        self.write_mesh(mesh)

        logger.info("Writing information")
        pg = mesh_data.physical_groups
        self.write_information(self.KEYS_OF_MESHTAGS, str(pg.keys()))

        logger.info("Writing cell meshtags")
        # TODO: former connectivity computation?
        if (ct := mesh_data.cell_tags) is not None:
            self.write_meshtags(ct, mesh.geometry)

        if (ft := mesh_data.facet_tags) is not None:
            self.write_meshtags(ft, mesh.geometry)

        if (rt := mesh_data.ridge_tags) is not None:
            self.write_meshtags(rt, mesh.geometry)

        if (pt := mesh_data.peak_tags) is not None:
            self.write_meshtags(pt, mesh.geometry)

    def read_mesh_meshtags(self, mesh_name="mesh"):
        """Read mesh and meshtags from XDMFFile.

        Parameters
        ----------
        mesh_name: optional
            Name of the Grid node containing the mesh data in XDMF file

        Returns
        -------
        mesh:
            The dolfin mesh
        mts:
            The dict of meshtags

        """
        logger = logging.getLogger("dolfiny")

        logger.info("Reading mesh")
        mesh = self.read_mesh(name=mesh_name)

        for d in range(mesh.topology.dim):
            mesh.topology.create_connectivity(d, mesh.topology.dim)

        logger.info("Reading information")
        value = self.read_information(self.KEYS_OF_MESHTAGS)

        import ast

        keys_meshtags = ast.literal_eval(value)

        mts = {}

        logger.info("Reading meshtags")
        for key in keys_meshtags.keys():
            mts[key] = self.read_meshtags(mesh, key)

        return mesh, mts

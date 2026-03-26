import gmsh
import numpy as np


def wireframe_mesh(model: gmsh.model, name: str = "wireframe") -> gmsh.model:
    """Extract from a gmsh model the wireframe/edge-only mesh.

    Adds (and selects) a new model containing the wireframe mesh. Physical groups are transferred
    from the input.

    Arguments:
        model: gmsh model
        name: name of the wireframe model

    Returns:
        wireframe model.

    """
    phys_edges = {}
    for dim, pg_tag in model.getPhysicalGroups():
        if dim == 0:
            continue

        edges = np.empty((0, 0), dtype=np.float64)
        for entity_tag in model.getEntitiesForPhysicalGroup(dim, pg_tag):
            for etype, _, ntags in zip(*model.mesh.getElements(dim=dim, tag=entity_tag)):
                assert dim > 0
                nodes = ntags if dim == 1 else model.mesh.getElementEdgeNodes(etype, tag=entity_tag)
                edges = np.append(edges, nodes)

        # sort and unique
        edges = edges.reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        phys_edges[(dim, pg_tag)] = (model.getPhysicalName(dim, pg_tag), edges)

    # store for all 0 dim PG the associated tags
    pg_nodes = {
        t: (model.getEntitiesForPhysicalGroup(d, t), model.getPhysicalName(d, t))
        for d, t in model.getPhysicalGroups(dim=0)
    }

    entities = {}
    for dimtag in model.getEntities():
        entities[dimtag] = (
            gmsh.model.getBoundary([dimtag]),
            gmsh.model.mesh.getNodes(*dimtag),
            gmsh.model.mesh.getElements(*dimtag),
        )

    model.add(name)

    for dimtag in sorted(entities):
        dim, tag = dimtag
        gmsh.model.addDiscreteEntity(dim, tag, [b[1] for b in entities[dimtag][0]])
        gmsh.model.mesh.addNodes(dim, tag, entities[dimtag][1][0], entities[dimtag][1][1])

        if dim > 0:
            continue

        assert len(entities[dimtag][2]) == 3
        gmsh.model.mesh.addElements(*dimtag, *entities[dimtag][2])

    for (_, tag), (phys_name, edges) in phys_edges.items():
        entity_tag = model.addDiscreteEntity(1)
        model.mesh.addElementsByType(entity_tag, 1, [], edges.flatten())

        new_tag = model.addPhysicalGroup(1, [entity_tag], tag=tag)
        if phys_name:
            model.setPhysicalName(1, new_tag, phys_name)

    for tag, (nodes, name) in pg_nodes.items():
        pg = model.addPhysicalGroup(0, nodes, tag=tag)
        model.setPhysicalName(0, pg, name)

    return model

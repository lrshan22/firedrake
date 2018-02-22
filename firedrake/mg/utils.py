import numpy
from pyop2 import op2
from pyop2.datatypes import IntType
from firedrake.functionspacedata import entity_dofs_key


def fine_node_to_coarse_node_map(Vf, Vc):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op2.MixedMap(fine_node_to_coarse_node_map(f, c) for f, c in zip(Vf, Vc))
    mesh = Vf.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    hierarchyf, levelf = get_level(Vf.ufl_domain())
    hierarchyc, levelc = get_level(Vc.ufl_domain())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    if levelc + 1 != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = (entity_dofs_key(Vc.finat_element.entity_dofs()) +
           entity_dofs_key(Vf.finat_element.entity_dofs()) +
           (levelc, levelf))

    cache = mesh._shared_data_cache["hierarchy_cell_node_map"]
    try:
        return cache[key]
    except KeyError:
        # XXX: Rewrite in cython.
        fine_to_coarse = hierarchy._fine_to_coarse[levelc+1]
        fine_map = Vf.cell_node_map()
        coarse_map = Vc.cell_node_map()
        fine_to_coarse_nodes = numpy.zeros((fine_map.toset.total_size,
                                            coarse_map.arity),
                                           dtype=IntType)
        for fcell, nodes in enumerate(fine_map.values_with_halo):
            ccell = fine_to_coarse[fcell]
            fine_to_coarse_nodes[nodes, :] = coarse_map.values_with_halo[ccell, :]

        return cache.setdefault(key, op2.Map(Vf.node_set, Vc.node_set, coarse_map.arity,
                                             values=fine_to_coarse_nodes))


def set_level(obj, hierarchy, level):
    """Attach hierarchy and level info to an object."""
    setattr(obj.topological, "__level_info__", (hierarchy, level))
    return obj


def get_level(obj):
    """Try and obtain hierarchy and level info from an object.

    If no level info is available, return ``None, None``."""
    try:
        return getattr(obj.topological, "__level_info__")
    except AttributeError:
        return None, None


def has_level(obj):
    """Does the provided object have level info?"""
    return hasattr(obj.topological, "__level_info__")

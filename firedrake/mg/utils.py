from pyop2 import op2
from firedrake.functionspacedata import entity_dofs_key


def coarse_to_fine_node_map(coarse, fine):
    if len(coarse) > 1:
        assert len(fine) == len(coarse)
        return op2.MixedMap(coarse_to_fine_node_map(c, f) for c, f in zip(coarse, fine))
    mesh = coarse.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    if not (coarse.ufl_element() == fine.ufl_element()):
        raise ValueError("Can't transfer between different spaces")
    ch, level = get_level(mesh)
    fh, fine_level = get_level(fine.mesh())
    if ch is not fh:
        raise ValueError("Can't map between different hierarchies")
    refinements_per_level = ch.refinements_per_level
    if refinements_per_level*level + 1 != refinements_per_level*fine_level:
        raise ValueError("Can't map between level %s and level %s" % (level, fine_level))
    c2f, vperm = ch._cells_vperm[int(level*refinements_per_level)]

    key = entity_dofs_key(coarse.finat_element.entity_dofs()) + (level, )
    cache = mesh._shared_data_cache["hierarchy_cell_node_map"]
    try:
        return cache[key]
    except KeyError:
        from .impl import create_cell_node_map
        map_vals, offset = create_cell_node_map(coarse, fine, c2f, vperm)
        return cache.setdefault(key, op2.Map(mesh.cell_set,
                                             fine.node_set,
                                             map_vals.shape[1],
                                             map_vals, offset=offset))


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

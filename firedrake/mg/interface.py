import numpy
from functools import partial

from pyop2 import op2
from pyop2.datatypes import IntType, as_cstr

import firedrake
import firedrake.utils
from . import utils


from ufl.algorithms.analysis import extract_arguments, extract_coefficients
from ufl.corealg.map_dag import map_expr_dags, map_expr_dag
import gem
import gem.impero_utils as impero_utils
import coffee.base as ast
import ufl
import tsfc
import tsfc.kernel_interface.firedrake as firedrake_interface
from tsfc.coffee import SCALAR_TYPE, generate as generate_coffee
from tsfc import ufl_utils
from tsfc.parameters import default_parameters


def to_reference_coordinates(ufl_coordinate_element, parameters=None):
    if parameters is None:
        parameters = tsfc.default_parameters()
    else:
        _ = tsfc.default_parameters()
        _.update(parameters)
        parameters = _

    def dX_norm_square(topological_dimension):
        return " + ".join("dX[{0}]*dX[{0}]".format(i)
                          for i in range(topological_dimension))

    def X_isub_dX(topological_dimension):
        return "\n".join("\tX[{0}] -= dX[{0}];".format(i)
                         for i in range(topological_dimension))

    def is_affine(ufl_element):
        return ufl_element.cell().is_simplex() and ufl_element.degree() <= 1 and ufl_element.family() in ["Discontinuous Lagrange", "Lagrange"]

    def init_X(fiat_cell):
        vertices = numpy.array(fiat_cell.get_vertices())
        X = numpy.average(vertices, axis=0)

        formatter = ast.ArrayInit(X, precision=parameters["precision"])._formatter
        return "\n".join("%s = %s;" % ("X[%d]" % i, formatter(v)) for i, v in enumerate(X))

    def to_reference_coordinates(ufl_coordinate_element):
        # Set up UFL form
        cell = ufl_coordinate_element.cell()
        domain = ufl.Mesh(ufl_coordinate_element)
        K = ufl.JacobianInverse(domain)
        x = ufl.SpatialCoordinate(domain)
        x0_element = ufl.VectorElement("Real", cell, 0)
        x0 = ufl.Coefficient(ufl.FunctionSpace(domain, x0_element))
        expr = ufl.dot(K, x - x0)

        # Translation to GEM
        C = ufl.Coefficient(ufl.FunctionSpace(domain, ufl_coordinate_element))
        expr = ufl_utils.preprocess_expression(expr)
        expr = ufl_utils.simplify_abs(expr)

        builder = firedrake_interface.KernelBuilderBase()
        builder.domain_coordinate[domain] = C
        builder._coefficient(C, "C")
        builder._coefficient(x0, "x0")

        dim = cell.topological_dimension()
        point = gem.Variable('X', (dim,))
        context = tsfc.fem.GemPointContext(
            interface=builder,
            ufl_cell=cell,
            precision=parameters["precision"],
            point_indices=(),
            point_expr=point,
        )
        translator = tsfc.fem.Translator(context)
        ir = map_expr_dag(translator, expr)

        # Unroll result
        ir = [gem.Indexed(ir, alpha) for alpha in numpy.ndindex(ir.shape)]

        # Unroll IndexSums
        max_extent = parameters["unroll_indexsum"]
        if max_extent:
            def predicate(index):
                return index.extent <= max_extent
            ir = gem.optimise.unroll_indexsum(ir, predicate=predicate)

        # Translate to COFFEE
        ir = impero_utils.preprocess_gem(ir)
        return_variable = gem.Variable('dX', (dim,))
        assignments = [(gem.Indexed(return_variable, (i,)), e)
                       for i, e in enumerate(ir)]
        impero_c = impero_utils.compile_gem(assignments, ())
        body = tsfc.coffee.generate(impero_c, {}, parameters["precision"])
        body.open_scope = False

        return body

    # Create FInAT element
    element = tsfc.finatinterface.create_element(ufl_coordinate_element)

    cell = ufl_coordinate_element.cell()

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "topological_dimension": cell.topological_dimension(),
        "to_reference_coords": to_reference_coordinates(ufl_coordinate_element),
        "init_X": init_X(element.cell),
        "max_iteration_count": 1 if is_affine(ufl_coordinate_element) else 16,
        "convergence_epsilon": 1e-12,
        "dX_norm_square": dX_norm_square(cell.topological_dimension()),
        "X_isub_dX": X_isub_dX(cell.topological_dimension()),
        "IntType": as_cstr(IntType),
    }

    evaluate_template_c = """#include <math.h>

static inline void to_reference_coords_kernel(double *X, const double *x0, const double *C)
{
    const int space_dim = %(geometric_dimension)d;

    /*
     * Mapping coordinates from physical to reference space
     */

%(init_X)s
    double x[space_dim];

    int converged = 0;
    for (int it = 0; !converged && it < %(max_iteration_count)d; it++) {
        double dX[%(topological_dimension)d] = { 0.0 };
%(to_reference_coords)s

        if (%(dX_norm_square)s < %(convergence_epsilon)g * %(convergence_epsilon)g) {
            converged = 1;
        }

%(X_isub_dX)s
    }
}"""

    return evaluate_template_c % code


def compile_element(expression, parameters=None):
    """Generates C code for point evaluations.
    :arg expression: UFL expression
    :arg coordinates: coordinate field
    :arg parameters: form compiler parameters
    :returns: C code as string
    """
    from tsfc.finatinterface import create_element
    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    # # No arguments, please!
    # if extract_arguments(expression):
    #     return ValueError("Cannot interpolate UFL expression with Arguments!")

    # Apply UFL preprocessing
    expression = tsfc.ufl_utils.preprocess_expression(expression)

    # # Collect required coefficients

    try:
        arg, = extract_coefficients(expression)
        argument_multiindex = ()
        coefficient = True
    except ValueError:
        arg, = extract_arguments(expression)
        finat_elem = create_element(arg.ufl_element())
        argument_multiindex = (finat_elem.get_indices(), )
        coefficient = False

    # # Point evaluation of mixed coefficients not supported here
    # if type(coefficient.ufl_element()) == MixedElement:
    #     raise NotImplementedError("Cannot point evaluate mixed elements yet!")

    # Replace coordinates (if any)
    builder = firedrake_interface.KernelBuilderBase()
    domain = expression.ufl_domain()
    # # Replace coordinates (if any)
    # domain = expression.ufl_domain()
    # assert coordinates.ufl_domain() == domain
    # expression = tsfc.ufl_utils.replace_coordinates(expression, coordinates)

    # Initialise kernel builder
    # f_arg = builder._coefficient(coefficient, "f")

    # TODO: restore this for expression evaluation!
    # expression = ufl_utils.split_coefficients(expression, builder.coefficient_split)

    # Translate to GEM
    cell = domain.ufl_cell()
    dim = cell.topological_dimension()
    point = gem.Variable('X', (dim,))
    point_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('X', rank=(dim,)))

    config = dict(interface=builder,
                  ufl_cell=cell,
                  precision=parameters["precision"],
                  point_indices=(),
                  point_expr=point,
                  argument_multiindices=argument_multiindex)
    # TODO: restore this for expression evaluation!
    context = tsfc.fem.GemPointContext(**config)

    # Abs-simplification
    expression = tsfc.ufl_utils.simplify_abs(expression)

    # Translate UFL -> GEM
    translator = tsfc.fem.Translator(context)
    result, = map_expr_dags(translator, [expression])

    tensor_indices = ()
    if coefficient:
        if expression.ufl_shape:
            tensor_indices = tuple(gem.Index() for s in expression.ufl_shape)
            return_variable = gem.Indexed(gem.Variable('R', expression.ufl_shape), tensor_indices)
            result_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('R', rank=expression.ufl_shape))
            result = gem.Indexed(result, tensor_indices)
        else:
            return_variable = gem.Indexed(gem.Variable('R', (1,)), (0,))
            result_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('R', rank=(1,)))
    else:
        return_variable = gem.Indexed(gem.Variable('R', finat_elem.index_shape), argument_multiindex)
        result_arg = ast.Decl(SCALAR_TYPE, ast.Symbol('R', rank=finat_elem.index_shape))

    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        result, = gem.optimise.unroll_indexsum([result], predicate=predicate)

    # Translate GEM -> COFFEE
    result, = gem.impero_utils.preprocess_gem([result])
    impero_c = gem.impero_utils.compile_gem([(return_variable, result)], tensor_indices)
    body = generate_coffee(impero_c, {}, parameters["precision"])

    # Build kernel tuple
    kernel_code = builder.construct_kernel("evaluate_kernel", [result_arg, point_arg], body)

    return kernel_code


def prolong_kernel(expression):
    from firedrake import TestFunction
    V = expression.function_space()
    coordinates = expression.ufl_domain().coordinates

    mesh = coordinates.ufl_domain()
    evaluate_kernel = compile_element(expression)
    to_reference_kernel = to_reference_coordinates(coordinates.ufl_element())

    my_kernel = """
    %(to_reference)s
    %(evaluate)s
    void kernel(double R[%(ndof)d], const double X[%(gdim)d], const double *Xc)
    {
        double Xref[%(tdim)d];
        to_reference_coords_kernel(Xref, X, Xc);
        evaluate_kernel(R, Xref);
    }
    """ % {"to_reference": str(to_reference_kernel),
           "evaluate": str(evaluate_kernel),
           "ndof": V.cell_node_map().arity * V.value_size,
           "gdim": mesh.geometric_dimension(),
           "tdim": mesh.topological_dimension()}

    return op2.Kernel(my_kernel, name="kernel")


__all__ = ["prolong", "restrict", "inject", "FunctionHierarchy",
           "FunctionSpaceHierarchy", "VectorFunctionSpaceHierarchy",
           "TensorFunctionSpaceHierarchy", "MixedFunctionSpaceHierarchy"]


def check_arguments(coarse, fine):
    cfs = coarse.function_space()
    ffs = fine.function_space()
    hierarchy, lvl = utils.get_level(cfs.mesh())
    if hierarchy is None:
        raise ValueError("Coarse function not from hierarchy")
    fhierarchy, flvl = utils.get_level(ffs.mesh())
    if lvl >= flvl:
        raise ValueError("Coarse function must be from coarser space")
    if hierarchy is not fhierarchy:
        raise ValueError("Can't transfer between functions from different hierarchies")



def prolong(input, output):

    coarse_V = input.function_space()
    fine_V = output.function_space()
    c2f_map = utils.coarse_to_fine_node_map(coarse_V, fine_V).values_with_halo

    coarse_coordinates = coarse_V.ufl_domain().coordinates
    d = {}
    coarse_map = coarse_V.cell_node_map().values_with_halo
    coords_map = coarse_coordinates.cell_node_map().values_with_halo
    dc = {}
    for cell, nodes in enumerate(c2f_map):
        for node in nodes:
            if node in d:
                continue
            d[node] = coarse_map[cell, :]
            dc[node] = coords_map[cell, :]

    map_from_fine_node_to_coarse_nodes = numpy.asarray([d[k] for k in sorted(d.keys())])

    map_from_fine_node_to_coarse_nodes = op2.Map(output.node_set, input.node_set, map_from_fine_node_to_coarse_nodes.shape[1],
                                                 values=map_from_fine_node_to_coarse_nodes)

    map_from_fine_node_to_coarse_coordinate_nodes = numpy.asarray([dc[k] for k in sorted(dc.keys())])
    map_from_fine_node_to_coarse_coordinate_nodes = op2.Map(output.node_set, coarse_coordinates.node_set, map_from_fine_node_to_coarse_coordinate_nodes.shape[1],
                                                            values=map_from_fine_node_to_coarse_coordinate_nodes)

    from IPython import embed; embed()
    op2.par_loop(kernel, output.node_set,
                 output.node_set(op2.WRITE),
                 input.node_set(op2.READ, map_from_fine_node_to_coarse_nodes),
                 input_node_physical_location(op2.READ),
                 coarse_coordinates(op2.READ, map_from_fine_node_to_coarse_coordinate_nodes))

    
@firedrake.utils.known_pyop2_safe
def transfer(input, output, typ=None):
    if len(input.function_space()) > 1:
        if len(output.function_space()) != len(input.function_space()):
            raise ValueError("Mixed spaces have different lengths")
        for in_, out in zip(input.split(), output.split()):
            transfer(in_, out, typ=typ)
        return

    if typ == "prolong":
        coarse, fine = input, output
    elif typ in ["inject", "restrict"]:
        coarse, fine = output, input
    else:
        raise ValueError("Unknown transfer type '%s'" % typ)
    check_arguments(coarse, fine)

    hierarchy, coarse_level = utils.get_level(coarse.ufl_domain())
    _, fine_level = utils.get_level(fine.ufl_domain())
    ref_per_level = hierarchy.refinements_per_level
    all_meshes = hierarchy._unskipped_hierarchy

    kernel = None
    element = input.ufl_element()
    repeat = (fine_level - coarse_level)*ref_per_level
    if typ == "prolong":
        next_level = coarse_level*ref_per_level
    else:
        next_level = fine_level*ref_per_level

    for j in range(repeat):
        if typ == "prolong":
            next_level += 1
        else:
            next_level -= 1
        if j == repeat - 1:
            next = output
        else:
            V = firedrake.FunctionSpace(all_meshes[next_level], element)
            next = firedrake.Function(V)
        if typ == "prolong":
            coarse, fine = input, next
        else:
            coarse, fine = next, input

        coarse_V = coarse.function_space()
        fine_V = fine.function_space()
        if kernel is None:
            kernel = utils.get_transfer_kernel(coarse_V, fine_V, typ=typ)
        c2f_map = utils.coarse_to_fine_node_map(coarse_V, fine_V)
        args = [kernel, c2f_map.iterset]
        if typ == "prolong":
            args.append(next.dat(op2.WRITE, c2f_map[op2.i[0]]))
            args.append(input.dat(op2.READ, input.cell_node_map()))
        elif typ == "inject":
            args.append(next.dat(op2.WRITE, next.cell_node_map()[op2.i[0]]))
            args.append(input.dat(op2.READ, c2f_map))
        else:
            next.dat.zero()
            args.append(next.dat(op2.INC, next.cell_node_map()[op2.i[0]]))
            args.append(input.dat(op2.READ, c2f_map))
            weights = utils.get_restriction_weights(coarse_V, fine_V)
            if weights is not None:
                args.append(weights.dat(op2.READ, c2f_map))

        op2.par_loop(*args)
        input = next


restrict = partial(transfer, typ="restrict")

inject = partial(transfer, typ="inject")


def FunctionHierarchy(fs_hierarchy, functions=None):
    """ outdated and returns warning & list of functions corresponding to each level
    of a functionspace hierarchy

        :arg fs_hierarchy: the :class:`~.FunctionSpaceHierarchy` to build on.
        :arg functions: optional :class:`~.Function` for each level.

    """
    from firedrake.logging import warning, RED
    warning(RED % "FunctionHierarchy is obsolete. Falls back by returning list of functions")

    if functions is not None:
        assert len(functions) == len(fs_hierarchy)
        for f, V in zip(functions, fs_hierarchy):
            assert f.function_space() == V
        return tuple(functions)
    else:
        return tuple([firedrake.Function(f) for f in fs_hierarchy])


def FunctionSpaceHierarchy(mesh_hierarchy, *args, **kwargs):
    from firedrake.logging import warning, RED
    warning(RED % "FunctionSpaceHierarchy is obsolete. Just build a FunctionSpace on the relevant mesh")

    return tuple(firedrake.FunctionSpace(mesh, *args, **kwargs) for mesh in mesh_hierarchy)


def VectorFunctionSpaceHierarchy(mesh_hierarchy, *args, **kwargs):
    from firedrake.logging import warning, RED
    warning(RED % "VectorFunctionSpaceHierarchy is obsolete. Just build a FunctionSpace on the relevant mesh")

    return tuple(firedrake.VectorFunctionSpace(mesh, *args, **kwargs) for mesh in mesh_hierarchy)


def TensorFunctionSpaceHierarchy(mesh_hierarchy, *args, **kwargs):
    from firedrake.logging import warning, RED
    warning(RED % "TensorFunctionSpaceHierarchy is obsolete. Just build a FunctionSpace on the relevant mesh")

    return tuple(firedrake.TensorFunctionSpace(mesh, *args, **kwargs) for mesh in mesh_hierarchy)


def MixedFunctionSpaceHierarchy(mesh_hierarchy, *args, **kwargs):
    from firedrake.logging import warning, RED
    warning(RED % "TensorFunctionSpaceHierarchy is obsolete. Just build a FunctionSpace on the relevant mesh")

    kwargs.pop("mesh", None)
    return tuple(firedrake.MixedFunctionSpace(*args, mesh=mesh, **kwargs) for mesh in mesh_hierarchy)

import numpy
from pyop2 import op2
from pyop2.datatypes import IntType, as_cstr
from firedrake.functionspacedata import entity_dofs_key
import firedrake

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


def compile_element(expression, dual_space=None, parameters=None,
                    name="evaluate_kernel"):
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
        argument_multiindices = ()
        coefficient = True
    except ValueError:
        arg, = extract_arguments(expression)
        finat_elem = create_element(arg.ufl_element())
        argument_multiindices = (finat_elem.get_indices(), )
        argument_multiindex, = argument_multiindices
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
                  argument_multiindices=argument_multiindices)
    # TODO: restore this for expression evaluation!
    context = tsfc.fem.GemPointContext(**config)

    # Abs-simplification
    expression = tsfc.ufl_utils.simplify_abs(expression)

    # Translate UFL -> GEM
    if coefficient:
        assert dual_space is None
        f_arg = [builder._coefficient(arg, "f")]
    else:
        f_arg = []
    translator = tsfc.fem.Translator(context)
    result, = map_expr_dags(translator, [expression])

    tensor_indices = ()
    b_arg = []
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
        if dual_space:
            elem = create_element(dual_space.ufl_element())
            if elem.value_shape:
                var = gem.Indexed(gem.Variable("b", elem.value_shape),
                                  elem.get_value_indices())
                b_arg = [ast.Decl(SCALAR_TYPE, ast.Symbol("b", rank=elem.value_shape))]
            else:
                var = gem.Indexed(gem.Variable("b", (1, )), (0, ))
                b_arg = [ast.Decl(SCALAR_TYPE, ast.Symbol("b", rank=(1, )))]
            result = gem.Product(result, var)

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
    kernel_code = builder.construct_kernel(name, [result_arg] + b_arg + f_arg + [point_arg], body)

    return kernel_code


def prolong_kernel(expression):
    cache = expression.ufl_domain()._shared_data_cache["transfer_kernels"]
    coordinates = expression.ufl_domain().coordinates
    key = (("prolong", ) +
           entity_dofs_key(expression.function_space().finat_element.entity_dofs()) +
           entity_dofs_key(coordinates.function_space().finat_element.entity_dofs()))
    try:
        return cache[key]
    except KeyError:
        mesh = coordinates.ufl_domain()
        evaluate_kernel = compile_element(expression)
        to_reference_kernel = to_reference_coordinates(coordinates.ufl_element())

        eval_args = evaluate_kernel.args[:-1]

        args = ", ".join(a.gencode(not_scope=True) for a in eval_args)
        arg_names = ", ".join(a.sym.symbol for a in eval_args)
        my_kernel = """
        %(to_reference)s
        %(evaluate)s
        #include <stdio.h>
        void prolong_kernel(%(args)s, const double *X, const double *Xc)
        {
            double Xref[%(tdim)d];
            to_reference_coords_kernel(Xref, X, Xc);
            evaluate_kernel(%(arg_names)s, Xref);
        }
        """ % {"to_reference": str(to_reference_kernel),
               "evaluate": str(evaluate_kernel),
               "args": args,
               "arg_names": arg_names,
               "tdim": mesh.topological_dimension()}

        return cache.setdefault(key, op2.Kernel(my_kernel, name="prolong_kernel"))


def restrict_kernel(Vf, Vc):
    cache = Vf.ufl_domain()._shared_data_cache["transfer_kernels"]
    coordinates = Vc.ufl_domain().coordinates
    key = (("restrict", ) +
           entity_dofs_key(Vf.finat_element.entity_dofs()) +
           entity_dofs_key(Vc.finat_element.entity_dofs()) +
           entity_dofs_key(coordinates.function_space().finat_element.entity_dofs()))
    try:
        return cache[key]
    except KeyError:
        mesh = coordinates.ufl_domain()
        evaluate_kernel = compile_element(firedrake.TestFunction(Vc), Vf)
        to_reference_kernel = to_reference_coordinates(coordinates.ufl_element())

        eval_args = evaluate_kernel.args[:-1]

        args = ", ".join(a.gencode(not_scope=True) for a in eval_args)
        arg_names = ", ".join(a.sym.symbol for a in eval_args)
        my_kernel = """
        %(to_reference)s
        %(evaluate)s
        #include <stdio.h>
        void restrict_kernel(%(args)s, const double *X, const double *Xc)
        {
            double Xref[%(tdim)d];
            to_reference_coords_kernel(Xref, X, Xc);
            evaluate_kernel(%(arg_names)s, Xref);
        }
        """ % {"to_reference": str(to_reference_kernel),
               "evaluate": str(evaluate_kernel),
               "args": args,
               "arg_names": arg_names,
               "tdim": mesh.topological_dimension()}

        return cache.setdefault(key, op2.Kernel(my_kernel, name="restrict_kernel"))

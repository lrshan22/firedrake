from firedrake import *
import pytest
import numpy as np
import itertools


def run_prolongation(mtype, vector, space, degree):
    if mtype == "interval":
        m = UnitIntervalMesh(10)
    elif mtype == "square":
        m = UnitSquareMesh(4, 4)
    nref = 2
    mh = MeshHierarchy(m, nref)

    mesh = mh[0]
    if vector:
        V = VectorFunctionSpace(mesh, space, degree)
        # Exactly represented on coarsest grid
        if mtype == "interval":
            expr = Expression(("pow(x[0], d)", ), d=degree)
        elif mtype == "square":
            expr = Expression(("pow(x[0], d) - pow(x[1], d)",
                               "pow(x[0], d) + pow(x[1], d)"), d=degree)
    else:
        V = FunctionSpace(mesh, space, degree)
        # Exactly represented on coarsest grid
        if mtype == "interval":
            expr = Expression("pow(x[0], d)", d=degree)
        elif mtype == "square":
            expr = Expression("pow(x[0], d) - pow(x[1], d)", d=degree)

    actual = Function(V).interpolate(expr)

    for mesh in mh[1:]:
        V = FunctionSpace(mesh, V.ufl_element())
        expect = Function(V).interpolate(expr)
        tmp = Function(V)
        prolong(actual, tmp)
        actual = tmp
        assert np.allclose(expect.dat.data_ro, actual.dat.data_ro)


@pytest.mark.parametrize(["mtype", "degree", "vector", "fs"],
                         itertools.product(("interval", "square"),
                                           range(0, 4),
                                           [False, True],
                                           ["CG", "DG"]))
def test_prolongation(mtype, degree, vector, fs):
    if fs == "CG" and degree == 0:
        pytest.skip("CG0 makes no sense")
    if fs == "DG" and degree == 3:
        pytest.skip("DG3 too expensive")
    run_prolongation(mtype, vector, fs, degree)


@pytest.mark.parallel(nprocs=2)
def test_cg_prolongation_square_parallel():
    for degree in range(1, 4):
        run_prolongation("square", False, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_dg_prolongation_square_parallel():
    for degree in range(0, 3):
        run_prolongation("square", False, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_cg_prolongation_square_parallel():
    for degree in range(1, 4):
        run_prolongation("square", True, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_dg_prolongation_square_parallel():
    for degree in range(0, 3):
        run_prolongation("square", True, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_cg_prolongation_interval_parallel():
    for degree in range(1, 4):
        run_prolongation("interval", False, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_dg_prolongation_interval_parallel():
    for degree in range(0, 3):
        run_prolongation("interval", False, "DG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_cg_prolongation_interval_parallel():
    for degree in range(1, 4):
        run_prolongation("interval", True, "CG", degree)


@pytest.mark.parallel(nprocs=2)
def test_vector_dg_prolongation_interval_parallel():
    for degree in range(0, 3):
        run_prolongation("interval", True, "DG", degree)


def run_extruded_prolongation(mtype, vector, space, degree):
    if mtype == "interval":
        m = UnitIntervalMesh(10)
    elif mtype == "square":
        m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    emh = ExtrudedMeshHierarchy(mh, layers=3)

    mesh = emh[0]
    if vector:
        V = VectorFunctionSpace(mesh, space, degree)
        # Exactly represented on coarsest grid
        if mtype == "interval":
            expr = Expression(("pow(x[0], d)", "pow(x[1], d)"), d=degree)
        elif mtype == "square":
            expr = Expression(("pow(x[0], d) - pow(x[1], d)",
                               "pow(x[0], d) + pow(x[1], d)",
                               "pow(x[2], d)"), d=degree)
    else:
        V = FunctionSpace(mesh, space, degree)
        # Exactly represented on coarsest grid
        expr = Expression("pow(x[0], d)", d=degree)

    actual = Function(V).interpolate(expr)

    for mesh in emh[1:]:
        V = FunctionSpace(mesh, V.ufl_element())
        expect = Function(V).interpolate(expr)
        tmp = Function(V)
        prolong(actual, tmp)
        actual = tmp
        assert np.allclose(expect.dat.data_ro, actual.dat.data_ro)


@pytest.mark.xfail(reason="Transfers not implemented in new manner")
@pytest.mark.parametrize(["mtype", "vector", "space", "degree"],
                         itertools.product(("interval", "square"),
                                           [False, True],
                                           ["CG", "DG"],
                                           range(0, 4)))
def test_extruded_prolongation(mtype, vector, space, degree):
    if space == "CG" and degree == 0:
        pytest.skip("CG0 makes no sense")
    if space == "DG" and degree == 3:
        pytest.skip("DG3 too expensive")
    run_extruded_prolongation(mtype, vector, space, degree)


@pytest.mark.xfail(reason="Transfers not implemented in new manner")
@pytest.mark.parallel(nprocs=2)
def test_extruded_dg_prolongation_square_parallel():
    for d in range(0, 3):
        run_extruded_prolongation("square", False, "DG", d)


@pytest.mark.xfail(reason="Transfers not implemented in new manner")
@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_dg_prolongation_square_parallel():
    for d in range(0, 3):
        run_extruded_prolongation("square", True, "DG", d)


@pytest.mark.xfail(reason="Transfers not implemented in new manner")
@pytest.mark.parallel(nprocs=2)
def test_extruded_cg_prolongation_square_parallel():
    for d in range(1, 4):
        run_extruded_prolongation("square", False, "CG", d)


@pytest.mark.xfail(reason="Transfers not implemented in new manner")
@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_cg_prolongation_square_parallel():
    for d in range(1, 4):
        run_extruded_prolongation("square", True, "CG", d)


@pytest.mark.xfail(reason="Transfers not implemented in new manner")
@pytest.mark.parallel(nprocs=2)
def test_extruded_dg_prolongation_interval_parallel():
    for d in range(0, 3):
        run_extruded_prolongation("interval", False, "DG", d)


@pytest.mark.xfail(reason="Transfers not implemented in new manner")
@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_dg_prolongation_interval_parallel():
    for d in range(0, 3):
        run_extruded_prolongation("interval", True, "DG", d)


@pytest.mark.xfail(reason="Transfers not implemented in new manner")
@pytest.mark.parallel(nprocs=2)
def test_extruded_cg_prolongation_interval_parallel():
    for d in range(1, 4):
        run_extruded_prolongation("interval", False, "CG", d)


@pytest.mark.xfail(reason="Transfers not implemented in new manner")
@pytest.mark.parallel(nprocs=2)
def test_extruded_vector_cg_prolongation_interval_parallel():
    for d in range(1, 4):
        run_extruded_prolongation("interval", True, "CG", d)


def run_mixed_prolongation():
    m = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(m, 2)

    mesh = mh[0]
    V = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)

    W = V*P

    expr = Expression(("x[0]*x[1]", "-x[1]*x[0]", "x[0] - x[1]"))

    actual = Function(W).interpolate(expr)

    for mesh in mh[1:]:
        W = FunctionSpace(mesh, W.ufl_element())
        expect = Function(W).interpolate(expr)
        tmp = Function(W)
        prolong(actual, tmp)
        actual = tmp
        for e, a in zip(expect.split(), actual.split()):
            assert np.allclose(e.dat.data_ro, a.dat.data_ro)


def test_mixed_prolongation():
    run_mixed_prolongation()


@pytest.mark.parallel(nprocs=2)
def test_mixed_prolongation_parallel():
    run_mixed_prolongation()


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))

from firedrake import *
import pytest


@pytest.mark.parametrize("options_prefix",
                         [None,
                          "",
                          "foo"])
def test_matrix_prefix_solver(options_prefix):
    parameters = {"ksp_type": "preonly",
                  "pc_type": "lu",
                  "pc_factor_mat_solver_package": "mumps",
                  "mat_mumps_icntl_24": 1}
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "P", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx
    L = v*dx
    uh = Function(V)

    problem = LinearVariationalProblem(a, L, uh)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters,
                                     options_prefix=options_prefix)
    solver.solve()

    pc = solver.snes.ksp.pc
    factor = pc.getFactorMatrix()
    assert factor.getType() == "mumps"
    assert factor.getMumpsIcntl(24) == 1

    for A in pc.getOperators():
        pfx = A.getOptionsPrefix()
        if pfx is None:
            pfx = ""
        assert pfx == solver.options_prefix

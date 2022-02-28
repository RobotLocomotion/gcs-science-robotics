import numpy as np
import pydot
import time
from scipy.optimize import root_scalar

from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    HPolyhedron,
    Point,
)
from pydrake.math import (
    BsplineBasis,
    BsplineBasis_,
    KnotVectorType,
)
from pydrake.solvers.mathematicalprogram import(
    Binding,
    CommonSolverOption,
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    PerspectiveQuadraticCost,
    SolverOptions,
)
from pydrake.solvers.mosek import MosekSolver
from pydrake.symbolic import (
    DecomposeLinearExpressions,
    Expression,
    MakeMatrixContinuousVariable,
    MakeVectorContinuousVariable,
)
from pydrake.trajectories import (
    BsplineTrajectory,
    BsplineTrajectory_,
    Trajectory,
)

from spp_helpers import findEdgesViaOverlaps, findStartGoalEdges, solveSPP

class BezierSPP:
    def __init__(self, regions, order, continuity, weights, deriv_limits=None, edges=None):
        self.dimension = regions[0].ambient_dimension()
        self.regions = regions
        self.order = order
        self.continuity = continuity
        self.weights = weights
        self.solver = MosekSolver()
        assert continuity < order
        for r in regions:
            assert r.ambient_dimension() == self.dimension
        if "time" in weights:
            assert isinstance(weights["time"], float) or isinstance(weights["time"], int)
        if "norm" in weights:
            assert len(weights["norm"]) < order + 1
        if "integral_norm" in weights:
            assert len(weights["integral_norm"]) < order + 1
        if "norm_squared" in weights:
            assert len(weights["norm_squared"]) < order + 1
            assert len(weights["norm_squared"]) <= 1
        if deriv_limits is not None:
            assert len(deriv_limits.shape) == 3
            assert deriv_limits.shape[0] == 1
            assert deriv_limits.shape[1] == 2
            assert deriv_limits.shape[2] == self.dimension

        self.spp = GraphOfConvexSets()

        A_time = np.vstack((np.eye(order + 1), -np.eye(order + 1),
                            np.eye(order, order + 1) - np.eye(order, order + 1, 1)))
        b_time = np.concatenate((1000*np.ones(order + 1), np.zeros(order + 1), -1e-6 * np.ones(order)))
        self.time_scaling_set = HPolyhedron(A_time, b_time)

        for r in regions:
            self.spp.AddVertex(
                r.CartesianPower(order + 1).CartesianProduct(self.time_scaling_set))

        # Formulate edge costs and constraints
        u_control = MakeMatrixContinuousVariable(
            self.dimension, order + 1, "xu")
        v_control = MakeMatrixContinuousVariable(
            self.dimension, order + 1, "xv")
        u_duration = MakeVectorContinuousVariable(order + 1, "Tu")
        v_duration = MakeVectorContinuousVariable(order + 1, "Tv")
        u_control_vars = []
        v_control_vars = []
        u_duration_vars = []
        v_duration_vars = []
        for ii in range(order + 1):
            u_control_vars.append(u_control[:, ii])
            v_control_vars.append(v_control[:, ii])
            u_duration_vars.append(np.array([u_duration[ii]]))
            v_duration_vars.append(np.array([v_duration[ii]]))
        u_vars = np.concatenate((u_control.flatten("F"), u_duration))
        edge_vars = np.concatenate((u_control.flatten("F"), u_duration, v_control.flatten("F"), v_duration))
        u_r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            u_control_vars)
        v_r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            v_control_vars)
        u_h_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            u_duration_vars)
        v_h_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            v_duration_vars)

        # Continuity constraints
        self.contin_constraints = []
        for deriv in range(continuity + 1):
            u_path_deriv = u_r_trajectory.MakeDerivative(deriv)
            v_path_deriv = v_r_trajectory.MakeDerivative(deriv)
            path_continuity_error = v_path_deriv.control_points()[0] - u_path_deriv.control_points()[-1]
            self.contin_constraints.append(LinearEqualityConstraint(
                DecomposeLinearExpressions(path_continuity_error, edge_vars),
                np.zeros(self.dimension)))

            u_time_deriv = u_h_trajectory.MakeDerivative(deriv)
            v_time_deriv = v_h_trajectory.MakeDerivative(deriv)
            time_continuity_error = v_time_deriv.control_points()[0] - u_time_deriv.control_points()[-1]
            self.contin_constraints.append(LinearEqualityConstraint(
                DecomposeLinearExpressions(time_continuity_error, edge_vars), 0.0))

        # Formulate derivative constraints
        self.deriv_constraints = []
        if deriv_limits is not None:
            u_path_control = u_r_trajectory.MakeDerivative(1).control_points()
            u_time_control = u_h_trajectory.MakeDerivative(1).control_points()
            lb = np.expand_dims(deriv_limits[0, 0], 1)
            ub = np.expand_dims(deriv_limits[0, 1], 1)

            for ii in range(len(u_path_control)):
                A_ctrl = DecomposeLinearExpressions(u_path_control[ii], u_vars)
                b_ctrl = DecomposeLinearExpressions(u_time_control[ii], u_vars)
                A_constraint = np.vstack((A_ctrl - ub * b_ctrl, -A_ctrl + lb * b_ctrl))
                self.deriv_constraints.append(LinearConstraint(
                    A_constraint, -np.inf*np.ones(2*self.dimension), np.zeros(2*self.dimension)))

        # Formulate path length and regularizing cost
        self.edge_costs = []

        if "time" in weights:
            u_time_control = u_h_trajectory.control_points()
            segment_time = u_time_control[-1] - u_time_control[0]
            self.edge_costs.append(LinearCost(
                weights["time"] * DecomposeLinearExpressions(segment_time, u_vars)[0], 0.))

        norm_weights = weights["norm"] if "norm" in weights else []
        for deriv in range(len(norm_weights)):
            if norm_weights[deriv] < 1e-12:
                continue
            u_path_control = u_r_trajectory.MakeDerivative(deriv + 1).control_points()
            for ii in range(len(u_path_control)):
                H = DecomposeLinearExpressions(u_path_control[ii] / (order - deriv), u_vars)
                self.edge_costs.append(L2NormCost(norm_weights[deriv] * H, np.zeros(self.dimension)))

        integral_weights = weights["integral_norm"] if "integral_norm" in weights else []
        integration_points = 100
        s_points = np.linspace(0., 1., integration_points + 1)
        for deriv in range(len(integral_weights)):
            if integral_weights[deriv] < 1e-12:
                continue
            u_path_deriv = u_r_trajectory.MakeDerivative(deriv + 1)
            q_ds = u_path_deriv.vector_values(s_points)

            if u_path_deriv.basis().order() == 1:
                for ii in [0, integration_points]:
                    costs = []
                    for jj in range(self.dimension):
                        costs.append(q_ds[jj, ii])
                    H = DecomposeLinearExpressions(costs, u_vars)
                    self.edge_costs.append(L2NormCost(integral_weights[deriv] * H, np.zeros(self.dimension)))
            else:
                for ii in range(integration_points + 1):
                    costs = []
                    for jj in range(self.dimension):
                        if ii == 0 or ii == integration_points:
                            costs.append(0.5 * 1./integration_points * q_ds[jj, ii])
                        else:
                            costs.append(1./integration_points * q_ds[jj, ii])
                    H = DecomposeLinearExpressions(costs, u_vars)
                    self.edge_costs.append(L2NormCost(integral_weights[deriv] * H, np.zeros(self.dimension)))

        norm_2_weights = weights["norm_squared"] if "norm_squared" in weights else []
        for deriv in range(len(norm_2_weights)):
            if norm_2_weights[deriv] < 1e-12:
                continue
            u_path_control = u_r_trajectory.MakeDerivative(deriv + 1).control_points()
            u_time_control = u_h_trajectory.MakeDerivative(deriv + 1).control_points()
            for ii in range(len(u_path_control)):
                A_ctrl = DecomposeLinearExpressions(u_path_control[ii], u_vars)
                b_ctrl = DecomposeLinearExpressions(u_time_control[ii], u_vars)
                H = np.vstack(((order) * b_ctrl, np.sqrt(norm_2_weights[deriv]) * A_ctrl))
                self.edge_costs.append(PerspectiveQuadraticCost(H))

        # Add edges to graph and apply costs/constraints
        if edges is None:
            edges = findEdgesViaOverlaps(self.regions)

        vertices = self.spp.Vertices()
        for ii, jj in edges:
            u = vertices[ii]
            v = vertices[jj]
            edge = self.spp.AddEdge(u, v, f"({ii}, {jj})")

            for cost in self.edge_costs:
                edge.AddCost(Binding[Cost](cost, u.x()))

            for c_con in self.contin_constraints:
                edge.AddConstraint(Binding[Constraint](
                        c_con, np.append(u.x(), v.x())))

            for d_con in self.deriv_constraints:
                edge.AddConstraint(Binding[Constraint](d_con, u.x()))

    def ResetGraph(self, vertices):
        for v in vertices:
            self.spp.RemoveVertex(v)
        for edge in self.spp.Edges():
            edge.ClearPhiConstraints()

    def VisualizeGraph(self):
        graphviz = self.spp.GetGraphvizString(None, False)
        return pydot.graph_from_dot_data(graphviz)[0].create_svg()

    def SolvePath(self, source, target, rounding=False, verbose=True, edges=None, velocity=None):
        assert len(source) == self.dimension
        assert len(target) == self.dimension
        if velocity is None:
            velocity = np.zeros((2, self.dimension))
        assert velocity.shape == (2, self.dimension)

        vertices = self.spp.Vertices()
        # Add edges connecting source and target to graph
        source_region = Point(source)
        target_region = Point(target)
        start = self.spp.AddVertex(source_region, "start")
        goal = self.spp.AddVertex(target_region, "goal")

        # Add edges connecting source and target to graph
        if edges is None:
            edges = findStartGoalEdges(self.regions, source, target)
        source_connected = (len(edges[0]) > 0)
        target_connected = (len(edges[1]) > 0)

        for ii in edges[0]:
            u = vertices[ii]
            edge = self.spp.AddEdge(start, u, f"(start, {ii})")

            for jj in range(self.dimension):
                edge.AddConstraint(start.x()[jj] == u.x()[jj])
                # edge.AddConstraint(
                #     u.x()[self.dimension + jj] - u.x()[jj]
                #         == velocity[0, jj] * (u.x()[-self.order] - u.x()[-(self.order + 1)]))
            edge.AddConstraint(u.x()[-(self.order + 1)] == 0.)

        for ii in edges[1]:
            u = vertices[ii]
            edge = self.spp.AddEdge(u, goal, f"({ii}, goal)")

            for cost in self.edge_costs:
                edge.AddCost(Binding[Cost](cost, u.x()))

            for jj in range(self.dimension):
                edge.AddConstraint(
                    u.x()[-(self.dimension + self.order + 1) + jj] == goal.x()[jj])
                # edge.AddConstraint(
                #     u.x()[-(self.dimension + self.order + 1) + jj] - u.x()[-(2*self.dimension + self.order + 1) + jj]
                #     == velocity[1, jj] * (u.x()[-1] - u.x()[-2]))

            for d_con in self.deriv_constraints:
                edge.AddConstraint(Binding[Constraint](d_con, u.x()))


        if not source_connected or not target_connected:
            print("Source connected:", source_connected, "Target connected:", target_connected)

        options = SolverOptions()
        options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
        options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
        options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
        # options.SetOption(GurobiSolver.id(), "MIPGap", 0.01)
        # options.SetOption(GurobiSolver.id(), "TimeLimit", 30.)

        active_edges, result, hard_result = solveSPP(
            self.spp, start, goal, rounding, self.solver, options)

        if verbose:
            print("Solution\t",
                  "Success:", result.get_solution_result(),
                  "Cost:", result.get_optimal_cost(),
                  "Solver time:", result.get_solver_details().optimizer_time)
            if rounding and hard_result is not None:
                print("Rounded Solution\t",
                      "Success:", hard_result.get_solution_result(),
                      "Cost:", hard_result.get_optimal_cost(),
                      "Solver time:",
                      hard_result.get_solver_details().optimizer_time)

        if active_edges is None:
            self.ResetGraph([start, goal])
            return None, result, hard_result

        if verbose:
            for edge in active_edges:
                print("Added", edge.name(), "to path. Value:",
                      result.GetSolution(edge.phi()))

        # Extract trajectory control points
        knots = np.zeros(self.order + 1)
        path_control_points = []
        time_control_points = []
        for edge in active_edges:
            if edge.v() == goal:
                knots = np.concatenate((knots, [knots[-1]]))
                path_control_points.append(hard_result.GetSolution(edge.xv()))
                time_control_points.append(np.array([hard_result.GetSolution(edge.xu())[-1]]))
                break
            edge_time = knots[-1] + 1.
            knots = np.concatenate((knots, np.full(self.order, edge_time)))
            edge_path_points = np.reshape(hard_result.GetSolution(edge.xv())[:-(self.order + 1)],
                                             (self.dimension, self.order + 1), "F")
            edge_time_points = hard_result.GetSolution(edge.xv())[-(self.order + 1):]
            for ii in range(self.order):
                path_control_points.append(edge_path_points[:, ii])
                time_control_points.append(np.array([edge_time_points[ii]]))

        offset = time_control_points[0].copy()
        for ii in range(len(time_control_points)):
            time_control_points[ii] -= offset

        path = BsplineTrajectory(BsplineBasis(self.order + 1, knots), path_control_points)
        time_traj = BsplineTrajectory(BsplineBasis(self.order + 1, knots), time_control_points)

        self.ResetGraph([start, goal])
        return BezierTrajectory(path, time_traj), result, hard_result

class BezierTrajectory(Trajectory):
    def __init__(self, path_traj, time_traj):
        assert path_traj.start_time() == time_traj.start_time()
        assert path_traj.end_time() == time_traj.end_time()
        self.path_traj = path_traj
        self.time_traj = time_traj
        self.start_s = path_traj.start_time()
        self.end_s = path_traj.end_time()

    def invert_time_traj(self, t):
        if t <= self.start_time():
            return self.start_s
        if t >= self.end_time():
            return self.end_s
        error = lambda s: self.time_traj.value(s)[0, 0] - t
        res = root_scalar(error, bracket=[self.start_s, self.end_s])
        return np.min([np.max([res.root, self.start_s]), self.end_s])

    def value(self, t):
        return self.path_traj.value(self.invert_time_traj(np.squeeze(t)))

    def vector_values(self, times):
        s = [self.invert_time_traj(t) for t in np.squeeze(times)]
        return self.path_traj.vector_values(s)

    def EvalDerivative(self, t, derivative_order=1):
        if derivative_order == 0:
            return self.value(t)
        elif derivative_order == 1:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            r_dot = self.path_traj.EvalDerivative(s, 1)
            return r_dot * s_dot
        elif derivative_order == 2:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            h_ddot = self.time_traj.EvalDerivative(s, 2)[0, 0]
            s_ddot = -h_ddot*(s_dot**3)
            r_dot = self.path_traj.EvalDerivative(s, 1)
            r_ddot = self.path_traj.EvalDerivative(s, 2)
            return r_ddot * s_dot * s_dot + r_dot * s_ddot
        else:
            raise ValueError()


    def start_time(self):
        return self.time_traj.value(self.start_s)[0, 0]

    def end_time(self):
        return self.time_traj.value(self.end_s)[0, 0]

    def rows(self):
        return self.path_traj.rows()

    def cols(self):
        return self.path_traj.cols()

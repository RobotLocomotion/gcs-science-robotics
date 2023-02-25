import numpy as np
import pydot
import time
from scipy.optimize import root_scalar

from pydrake.geometry.optimization import (
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
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    QuadraticCost,
    PerspectiveQuadraticCost,
)
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

from gcs.base import BaseGCS

class BezierGCS(BaseGCS):
    def __init__(self, regions, order, continuity, edges=None, hdot_min=1e-6, full_dim_overlap=False):
        BaseGCS.__init__(self, regions)

        self.order = order
        self.continuity = continuity
        assert continuity < order

        A_time = np.vstack((np.eye(order + 1), -np.eye(order + 1),
                            np.eye(order, order + 1) - np.eye(order, order + 1, 1)))
        b_time = np.concatenate((1e3*np.ones(order + 1), np.zeros(order + 1), -hdot_min * np.ones(order)))
        self.time_scaling_set = HPolyhedron(A_time, b_time)

        for i, r in enumerate(self.regions):
            self.gcs.AddVertex(
                r.CartesianPower(order + 1).CartesianProduct(self.time_scaling_set),
                name = self.names[i] if not self.names is None else '')

        # Formulate edge costs and constraints
        u_control = MakeMatrixContinuousVariable(
            self.dimension, order + 1, "xu")
        v_control = MakeMatrixContinuousVariable(
            self.dimension, order + 1, "xv")
        u_duration = MakeVectorContinuousVariable(order + 1, "Tu")
        v_duration = MakeVectorContinuousVariable(order + 1, "Tv")

        self.u_vars = np.concatenate((u_control.flatten("F"), u_duration))
        self.u_r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            u_control)
        self.u_h_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            np.expand_dims(u_duration, 0))

        edge_vars = np.concatenate((u_control.flatten("F"), u_duration, v_control.flatten("F"), v_duration))
        v_r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            v_control)
        v_h_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            np.expand_dims(v_duration, 0))

        # Continuity constraints
        self.contin_constraints = []
        for deriv in range(continuity + 1):
            u_path_deriv = self.u_r_trajectory.MakeDerivative(deriv)
            v_path_deriv = v_r_trajectory.MakeDerivative(deriv)
            path_continuity_error = v_path_deriv.control_points()[0] - u_path_deriv.control_points()[-1]
            self.contin_constraints.append(LinearEqualityConstraint(
                DecomposeLinearExpressions(path_continuity_error, edge_vars),
                np.zeros(self.dimension)))

            u_time_deriv = self.u_h_trajectory.MakeDerivative(deriv)
            v_time_deriv = v_h_trajectory.MakeDerivative(deriv)
            time_continuity_error = v_time_deriv.control_points()[0] - u_time_deriv.control_points()[-1]
            self.contin_constraints.append(LinearEqualityConstraint(
                DecomposeLinearExpressions(time_continuity_error, edge_vars), 0.0))

        self.deriv_constraints = []
        self.edge_costs = []

        # Add edges to graph and apply costs/constraints
        if edges is None:
            if full_dim_overlap:
                edges = self.findEdgesViaFullDimensionOverlaps()
            else:
                edges = self.findEdgesViaOverlaps()

        vertices = self.gcs.Vertices()
        for ii, jj in edges:
            u = vertices[ii]
            v = vertices[jj]
            edge = self.gcs.AddEdge(u, v, f"({u.name()}, {v.name()})")

            for c_con in self.contin_constraints:
                edge.AddConstraint(Binding[Constraint](
                        c_con, np.append(u.x(), v.x())))

    def addTimeCost(self, weight):
        assert isinstance(weight, float) or isinstance(weight, int)

        u_time_control = self.u_h_trajectory.control_points()
        segment_time = u_time_control[-1] - u_time_control[0]
        time_cost = LinearCost(
            weight * DecomposeLinearExpressions(segment_time, self.u_vars)[0], 0.)
        self.edge_costs.append(time_cost)

        for edge in self.gcs.Edges():
            if edge.u() == self.source:
                continue
            edge.AddCost(Binding[Cost](time_cost, edge.xu()))

    def addPathLengthCost(self, weight):
        if isinstance(weight, float) or isinstance(weight, int):
            weight_matrix = weight * np.eye(self.dimension)
        else:
            assert(len(weight) == self.dimension)
            weight_matrix = np.diag(weight)

        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        for ii in range(len(u_path_control)):
            H = DecomposeLinearExpressions(u_path_control[ii] / self.order, self.u_vars)
            path_cost = L2NormCost(np.matmul(weight_matrix, H), np.zeros(self.dimension))
            self.edge_costs.append(path_cost)

            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddCost(Binding[Cost](path_cost, edge.xu()))

    def addPathLengthIntegralCost(self, weight, integration_points=100):
        if isinstance(weight, float) or isinstance(weight, int):
            weight_matrix = weight * np.eye(self.dimension)
        else:
            assert(len(weight) == self.dimension)
            weight_matrix = np.diag(weight)

        s_points = np.linspace(0., 1., integration_points + 1)
        u_path_deriv = self.u_r_trajectory.MakeDerivative(1)

        if u_path_deriv.basis().order() == 1:
            for t in [0.0, 1.0]:
                q_ds = u_path_deriv.value(t)
                costs = []
                for ii in range(self.dimension):
                    costs.append(q_ds[ii])
                H = DecomposeLinearExpressions(costs, self.u_vars)
                integral_cost = L2NormCost(np.matmul(weight_matrix, H), np.zeros(self.dimension))
                self.edge_costs.append(integral_cost)

                for edge in self.gcs.Edges():
                    if edge.u() == self.source:
                        continue
                    edge.AddCost(Binding[Cost](integral_cost, edge.xu()))
        else:
            q_ds = u_path_deriv.vector_values(s_points)
            for ii in range(integration_points + 1):
                costs = []
                for jj in range(self.dimension):
                    if ii == 0 or ii == integration_points:
                        costs.append(0.5 * 1./integration_points * q_ds[jj, ii])
                    else:
                        costs.append(1./integration_points * q_ds[jj, ii])
                H = DecomposeLinearExpressions(costs, self.u_vars)
                integral_cost = L2NormCost(np.matmul(weight_matrix, H), np.zeros(self.dimension))
                self.edge_costs.append(integral_cost)

                for edge in self.gcs.Edges():
                    if edge.u() == self.source:
                        continue
                    edge.AddCost(Binding[Cost](integral_cost, edge.xu()))

    def addPathEnergyCost(self, weight):
        if isinstance(weight, float) or isinstance(weight, int):
            weight_matrix = weight * np.eye(self.dimension)
        else:
            assert(len(weight) == self.dimension)
            weight_matrix = np.diag(weight)

        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        for ii in range(len(u_path_control)):
            A_ctrl = DecomposeLinearExpressions(u_path_control[ii], self.u_vars)
            b_ctrl = DecomposeLinearExpressions(u_time_control[ii], self.u_vars)
            H = np.vstack(((self.order) * b_ctrl, np.matmul(np.sqrt(weight_matrix), A_ctrl)))
            energy_cost = PerspectiveQuadraticCost(H, np.zeros(H.shape[0]))
            self.edge_costs.append(energy_cost)

            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddCost(Binding[Cost](energy_cost, edge.xu()))

    def addDerivativeRegularization(self, weight_r, weight_h, order):

        assert isinstance(order, int) and 2 <= order <= self.order
        weights = [weight_r, weight_h]
        for weight in weights:
            assert isinstance(weight, float) or isinstance(weight, int)

        trajectories = [self.u_r_trajectory, self.u_h_trajectory]
        for traj, weight in zip(trajectories, weights):
            derivative_control = traj.MakeDerivative(order).control_points()
            for c in derivative_control:
                A_ctrl = DecomposeLinearExpressions(c, self.u_vars)
                H = A_ctrl.T.dot(A_ctrl) * 2 * weight / (1 + self.order - order)
                reg_cost = QuadraticCost(H, np.zeros(H.shape[0]), 0)
                self.edge_costs.append(reg_cost)

                for edge in self.gcs.Edges():
                    if edge.u() == self.source:
                        continue
                    edge.AddCost(Binding[Cost](reg_cost, edge.xu()))

    def addVelocityLimits(self, lower_bound, upper_bound):
        assert len(lower_bound) == self.dimension
        assert len(upper_bound) == self.dimension

        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        lb = np.expand_dims(lower_bound, 1)
        ub = np.expand_dims(upper_bound, 1)

        for ii in range(len(u_path_control)):
            A_ctrl = DecomposeLinearExpressions(u_path_control[ii], self.u_vars)
            b_ctrl = DecomposeLinearExpressions(u_time_control[ii], self.u_vars)
            A_constraint = np.vstack((A_ctrl - ub * b_ctrl, -A_ctrl + lb * b_ctrl))
            velocity_con = LinearConstraint(
                A_constraint, -np.inf*np.ones(2*self.dimension), np.zeros(2*self.dimension))
            self.deriv_constraints.append(velocity_con)

            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddConstraint(Binding[Constraint](velocity_con, edge.xu()))

    def addSourceTarget(self, source, target, edges=None, velocity=None, zero_deriv_boundary=None):
        source_edges, target_edges =  super().addSourceTarget(source, target, edges)

        if velocity is not None:
            assert velocity.shape == (2, self.dimension)

            u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
            u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
            initial_velocity_error = np.squeeze(u_path_control[0]) - velocity[0] * np.squeeze(u_time_control[0])
            final_velocity_error = np.squeeze(u_path_control[-1]) - velocity[1] * np.squeeze(u_time_control[-1])
            initial_velocity_con = LinearEqualityConstraint(
                DecomposeLinearExpressions(initial_velocity_error, self.u_vars),
                np.zeros(self.dimension))
            final_velocity_con = LinearEqualityConstraint(
                DecomposeLinearExpressions(final_velocity_error, self.u_vars),
                np.zeros(self.dimension))

        if zero_deriv_boundary is not None:
            assert self.order > zero_deriv_boundary + 1
            initial_constraints = []
            final_constraints = []

            for deriv in range(1, zero_deriv_boundary+1):
                u_path_control = self.u_r_trajectory.MakeDerivative(deriv).control_points()
                initial_constraints.append(LinearEqualityConstraint(
                    DecomposeLinearExpressions(np.squeeze(u_path_control[0]), self.u_vars),
                    np.zeros(self.dimension)))
                final_constraints.append(LinearEqualityConstraint(
                    DecomposeLinearExpressions(np.squeeze(u_path_control[-1]), self.u_vars),
                    np.zeros(self.dimension)))

        for edge in source_edges:
            for jj in range(self.dimension):
                edge.AddConstraint(edge.xu()[jj] == edge.xv()[jj])

            if velocity is not None:
                edge.AddConstraint(Binding[Constraint](initial_velocity_con, edge.xv()))
            if zero_deriv_boundary is not None:
                for i_con in initial_constraints:
                    edge.AddConstraint(Binding[Constraint](i_con, edge.xv()))

            edge.AddConstraint(edge.xv()[-(self.order + 1)] == 0.)

        for edge in target_edges:    
            for jj in range(self.dimension):
                edge.AddConstraint(
                    edge.xu()[-(self.dimension + self.order + 1) + jj] == edge.xv()[jj])

            if velocity is not None:
                edge.AddConstraint(Binding[Constraint](final_velocity_con, edge.xu()))
            if zero_deriv_boundary is not None:
                for f_con in final_constraints:
                    edge.AddConstraint(Binding[Constraint](f_con, edge.xu()))

            for cost in self.edge_costs:
                edge.AddCost(Binding[Cost](cost, edge.xu()))

            for d_con in self.deriv_constraints:
                edge.AddConstraint(Binding[Constraint](d_con, edge.xu()))


    def SolvePath(self, rounding=False, verbose=False, preprocessing=False):
        best_path, best_result, results_dict = self.solveGCS(
            rounding, preprocessing, verbose)

        if best_path is None:
            return None, results_dict

        # Extract trajectory control points
        knots = np.zeros(self.order + 1)
        path_control_points = []
        time_control_points = []
        for edge in best_path:
            if edge.v() == self.target:
                knots = np.concatenate((knots, [knots[-1]]))
                path_control_points.append(best_result.GetSolution(edge.xv()))
                time_control_points.append(np.array([best_result.GetSolution(edge.xu())[-1]]))
                break
            edge_time = knots[-1] + 1.
            knots = np.concatenate((knots, np.full(self.order, edge_time)))
            edge_path_points = np.reshape(best_result.GetSolution(edge.xv())[:-(self.order + 1)],
                                             (self.dimension, self.order + 1), "F")
            edge_time_points = best_result.GetSolution(edge.xv())[-(self.order + 1):]
            for ii in range(self.order):
                path_control_points.append(edge_path_points[:, ii])
                time_control_points.append(np.array([edge_time_points[ii]]))

        offset = time_control_points[0].copy()
        for ii in range(len(time_control_points)):
            time_control_points[ii] -= offset

        path_control_points = np.array(path_control_points).T
        time_control_points = np.array(time_control_points).T

        path = BsplineTrajectory(BsplineBasis(self.order + 1, knots), path_control_points)
        time_traj = BsplineTrajectory(BsplineBasis(self.order + 1, knots), time_control_points)

        return BezierTrajectory(path, time_traj), results_dict

class BezierTrajectory:
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

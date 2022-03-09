import numpy as np
import pydot
import time

from pydrake.geometry.optimization import (
    Point,
)
from pydrake.solvers.mathematicalprogram import (
    Binding,
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,
)

from spp.base import BaseSPP

class LinearSPP(BaseSPP):
    def __init__(self, regions, edges=None):
        BaseSPP.__init__(self, regions)

        self.edge_cost = L2NormCost(
            np.hstack((-np.eye(self.dimension), np.eye(self.dimension))),
            np.zeros(self.dimension))

        for i, r in enumerate(self.regions):
            self.spp.AddVertex(r, name = self.names[i] if not self.names is None else '')

        if edges is None:
            edges = self.findEdgesViaOverlaps()

        vertices = self.spp.Vertices()
        for ii, jj in edges:
            u = vertices[ii]
            v = vertices[jj]
            edge = self.spp.AddEdge(u, v, f"({u.name()}, {v.name()})")

            edge.AddCost(Binding[Cost](self.edge_cost, np.append(u.x(), v.x())))

            # Constrain point in v to be in u
            edge.AddConstraint(Binding[Constraint](
                LinearConstraint(u.set().A(),
                                 -np.inf*np.ones(len(u.set().b())),
                                 u.set().b()),
                v.x()))

    def SolvePath(self, source, target, rounding=False, verbose=False, edges=None):
        assert len(source) == self.dimension
        assert len(target) == self.dimension

        # Add source and target vertices
        vertices = self.spp.Vertices()
        start = self.spp.AddVertex(Point(source), "start")
        goal = self.spp.AddVertex(Point(target), "goal")

        # Add edges connecting source and target to graph
        if edges is None:
            edges = self.findStartGoalEdges(source, target)
        source_connected = (len(edges[0]) > 0)
        target_connected = (len(edges[1]) > 0)

        for ii in edges[0]:
            u = vertices[ii]
            edge = self.spp.AddEdge(start, u, f"(start, {u.name()})")

            for jj in range(self.dimension):
                edge.AddConstraint(start.x()[jj] == u.x()[jj])

        for ii in edges[1]:
            u = vertices[ii]
            edge = self.spp.AddEdge(u, goal, f"({u.name()}, goal)")

            edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(u.x(), goal.x())))

        if not source_connected or not target_connected:
            print("Source connected:", source_connected, "Target connected:", target_connected)

        active_edges, result, hard_result = self.solveSPP(start, goal, rounding)

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

        # Extract trajectory
        waypoints = np.empty((self.dimension, 0))
        for edge in active_edges:
            new_waypoint = hard_result.GetSolution(edge.xv())
            waypoints = np.concatenate(
                [waypoints, np.expand_dims(new_waypoint, 1)], axis=1)

        self.ResetGraph([start, goal])
        return waypoints, result, hard_result

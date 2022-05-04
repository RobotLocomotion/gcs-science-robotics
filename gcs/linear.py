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

from gcs.base import BaseGCS

class LinearGCS(BaseGCS):
    def __init__(self, regions, edges=None, path_weights=None, full_dim_overlap=False):
        BaseGCS.__init__(self, regions)

        if path_weights is None:
            path_weights = np.ones(self.dimension)
        assert len(path_weights) == self.dimension

        self.edge_cost = L2NormCost(
            np.hstack((np.diag(-path_weights), np.diag(path_weights))),
            np.zeros(self.dimension))

        for i, r in enumerate(self.regions):
            self.gcs.AddVertex(r, name = self.names[i] if not self.names is None else '')

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

            edge_length = edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(u.x(), v.x())))[1]
            self.edge_cost_dict[edge.id()] = [edge_length]

            # Constrain point in v to be in u
            edge.AddConstraint(Binding[Constraint](
                LinearConstraint(u.set().A(),
                                 -np.inf*np.ones(len(u.set().b())),
                                 u.set().b()),
                v.x()))

    def SolvePath(self, source, target, rounding=False, verbose=False, edges=None, preprocessing=False):
        assert len(source) == self.dimension
        assert len(target) == self.dimension

        # Add source and target vertices
        vertices = self.gcs.Vertices()
        start = self.gcs.AddVertex(Point(source), "start")
        goal = self.gcs.AddVertex(Point(target), "goal")

        # Add edges connecting source and target to graph
        if edges is None:
            edges = self.findStartGoalEdges(source, target)
        source_connected = (len(edges[0]) > 0)
        target_connected = (len(edges[1]) > 0)

        for ii in edges[0]:
            u = vertices[ii]
            edge = self.gcs.AddEdge(start, u, f"(start, {u.name()})")
            self.edge_cost_dict[edge.id()] = []

            for jj in range(self.dimension):
                edge.AddConstraint(start.x()[jj] == u.x()[jj])

        for ii in edges[1]:
            u = vertices[ii]
            edge = self.gcs.AddEdge(u, goal, f"({u.name()}, goal)")

            edge_length = edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(u.x(), goal.x())))[1]
            self.edge_cost_dict[edge.id()] = [edge_length]

        if not source_connected:
            raise ValueError('Source vertex is not connected.')
        if not target_connected:
            raise ValueError('Target vertex is not connected.')

        best_path, best_result, results_dict = self.solveGCS(
            start, goal, rounding, preprocessing, verbose)

        if best_path is None:
            self.ResetGraph([start, goal])
            return None, results_dict

        # Extract trajectory
        waypoints = np.empty((self.dimension, 0))
        for edge in best_path:
            new_waypoint = best_result.GetSolution(edge.xv())
            waypoints = np.concatenate(
                [waypoints, np.expand_dims(new_waypoint, 1)], axis=1)

        self.ResetGraph([start, goal])
        return waypoints, results_dict

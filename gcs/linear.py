import numpy as np
import pydot
import time

from pydrake.geometry.optimization import (
    Point,
)
from pydrake.solvers import (
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
        elif isinstance(path_weights, float) or isinstance(path_weights, int):
            path_weights = path_weights * np.ones(self.dimension)
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

            # Constrain point in v to be in u
            edge.AddConstraint(Binding[Constraint](
                LinearConstraint(u.set().A(),
                                 -np.inf*np.ones(len(u.set().b())),
                                 u.set().b()),
                v.x()))

    def addSourceTarget(self, source, target, edges=None):
        source_edges, target_edges = super().addSourceTarget(source, target, edges)

        for edge in source_edges:
            for jj in range(self.dimension):
                edge.AddConstraint(edge.xu()[jj] == edge.xv()[jj])

        for edge in target_edges:
            edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(edge.xu(), edge.xv())))


    def SolvePath(self, rounding=False, verbose=False, preprocessing=False):
        best_path, best_result, results_dict = self.solveGCS(
            rounding, preprocessing, verbose)

        if best_path is None:
            return None, results_dict

        # Extract trajectory
        waypoints = np.empty((self.dimension, 0))
        for edge in best_path:
            new_waypoint = best_result.GetSolution(edge.xv())
            waypoints = np.concatenate(
                [waypoints, np.expand_dims(new_waypoint, 1)], axis=1)

        return waypoints, results_dict

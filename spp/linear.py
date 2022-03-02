import numpy as np
import pydot
import time

from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    Point,
)
from pydrake.solvers.mathematicalprogram import (
    Binding,
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,
)

from spp.spp_helpers import (
    findEdgesViaOverlaps,
    findStartGoalEdges,
    greedyForwardPathSearch,
    solveSPP,
)

class LinearSPP:
    def __init__(self, regions, edges=None):
        self.dimension = regions[0].ambient_dimension()
        self.regions = regions.copy()
        self.solver = None
        self.options = None
        self.rounding_fn = greedyForwardPathSearch
        for r in self.regions:
            assert r.ambient_dimension() == self.dimension

        self.spp = GraphOfConvexSets()
        self.edge_cost = L2NormCost(
            np.hstack((-np.eye(self.dimension), np.eye(self.dimension))),
            np.zeros(self.dimension))

        for r in self.regions:
            self.spp.AddVertex(r)

        if edges is None:
            edges = findEdgesViaOverlaps(self.regions)

        vertices = self.spp.Vertices()
        for ii, jj in edges:
            u = vertices[ii]
            v = vertices[jj]
            edge = self.spp.AddEdge(u, v, f"({ii}, {jj})")

            edge.AddCost(Binding[Cost](self.edge_cost, np.append(u.x(), v.x())))

            # Constrain point in v to be in u
            edge.AddConstraint(Binding[Constraint](
                LinearConstraint(u.set().A(),
                                 -np.inf*np.ones(len(u.set().b())),
                                 u.set().b()),
                v.x()))

    def setSolver(self, solver):
        self.solver = solver

    def setSolverOptions(self, options):
        self.options = options

        # options = SolverOptions()
        # # options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # options.SetOption(GurobiSolver.id(), "MIPGap", 0.01)
        # options.SetOption(GurobiSolver.id(), "TimeLimit", 30.)

    def setRoundingStrategy(self, rounding_fn):
        self.rounding_fn = rounding_fn

    def ResetGraph(self, vertices):
        for v in vertices:
            self.spp.RemoveVertex(v)
        for edge in self.spp.Edges():
            edge.ClearPhiConstraints()

    def VisualizeGraph(self):
        graphviz = self.spp.GetGraphvizString(None, False)
        return pydot.graph_from_dot_data(graphviz)[0].create_svg()

    def SolvePath(self, source, target, rounding=False, verbose=False, edges=None):
        assert len(source) == self.dimension
        assert len(target) == self.dimension

        # Add source and target vertices
        vertices = self.spp.Vertices()
        start = self.spp.AddVertex(Point(source), "start")
        goal = self.spp.AddVertex(Point(target), "goal")

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

        for ii in edges[1]:
            u = vertices[ii]
            edge = self.spp.AddEdge(u, goal, f"({ii}, goal)")

            edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(u.x(), goal.x())))

        if not source_connected or not target_connected:
            print("Source connected:", source_connected, "Target connected:", target_connected)

        active_edges, result, hard_result = solveSPP(
            self.spp, start, goal, rounding, self.solver, self.options, self.rounding_fn)

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

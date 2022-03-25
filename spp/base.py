import pydot

from pydrake.geometry.optimization import (
    GraphOfConvexSets,
)
from pydrake.solvers.mathematicalprogram import (
    CommonSolverOption,
    SolverOptions,
)
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.solvers.mosek import MosekSolver

from spp.rounding import (
    greedyForwardPathSearch,
)

class BaseSPP:
    def __init__(self, regions):
        self.names = None
        if type(regions) is dict:
            self.names = list(regions.keys())
            regions = list(regions.values())
        else:
            self.names = ["v" + str(ii) for ii in range(len(regions))]
        self.dimension = regions[0].ambient_dimension()
        self.regions = regions.copy()
        self.solver = None
        self.options = None
        self.rounding_fn = [greedyForwardPathSearch]
        for r in self.regions:
            assert r.ambient_dimension() == self.dimension

        self.spp = GraphOfConvexSets()


    def findEdgesViaOverlaps(self):
        edges = []
        for ii in range(len(self.regions)):
            for jj in range(ii + 1, len(self.regions)):
                if self.regions[ii].IntersectsWith(self.regions[jj]):
                    edges.append((ii, jj))
                    edges.append((jj, ii))
        return edges

    def findStartGoalEdges(self, start, goal):
        edges = [[], []]
        for ii in range(len(self.regions)):
            if self.regions[ii].PointInSet(start):
                edges[0].append(ii)
            if self.regions[ii].PointInSet(goal):
                edges[1].append(ii)
        return edges

    def setSolver(self, solver):
        self.solver = solver

    def setSolverOptions(self, options):
        self.options = options

    def setPaperSolverOptions(self):
        self.options = SolverOptions()
        self.options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        self.options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
        self.options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
        self.options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
        self.options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", 3600.0)
        self.options.SetOption(GurobiSolver.id(), "MIPGap", 1e-3)
        self.options.SetOption(GurobiSolver.id(), "TimeLimit", 3600.0)

    def setRoundingStrategy(self, rounding_fn):
        if callable(rounding_fn):
            self.rounding_fn = [rounding_fn]
        elif isinstance(rounding_fn, list):
            assert len(rounding_fn) > 0
            for fn in rounding_fn:
                assert callable(fn)
            self.rounding_fn = rounding_fn
        else:
            raise ValueError("Rounding strategy must either be "
                             "a function or list of functions.")

    def ResetGraph(self, vertices):
        for v in vertices:
            self.spp.RemoveVertex(v)
        for edge in self.spp.Edges():
            edge.ClearPhiConstraints()

    def VisualizeGraph(self, file_type="svg"):
        graphviz = self.spp.GetGraphvizString(None, False)
        data = pydot.graph_from_dot_data(graphviz)[0]
        if file_type == "svg":
            return data.create_svg()
        elif file_type == "png":
            return data.create_png()
        else:
            raise ValueError("Unrecognized file type:", file_type)


    def solveSPP(self, start, goal, rounding):
        result = self.spp.SolveShortestPath(start, goal, rounding, self.solver, self.options)
        if not result.is_success():
            print("First solve failed")
            return None, result, None

        # Extract path
        active_edges = []
        found_path = False
        for fn in self.rounding_fn:
            rounded_edges = fn(self.spp, result, start, goal)
            if rounded_edges is None:
                print(fn.__name__, "could not find a path.")
            else:
                found_path = True
            active_edges.append(rounded_edges)
        if not found_path:
            print("All rounding strategies failed to find a path.")
            return None, result, None

        # Solve with hard edge choices
        if rounding:
            hard_result = []
            found_solution = False
            for path_edges in active_edges:
                if path_edges is None:
                    hard_result.append(None)
                    continue
                for edge in self.spp.Edges():
                    if edge in path_edges:
                        edge.AddPhiConstraint(True)
                    else:
                        edge.AddPhiConstraint(False)
                hard_result.append(self.spp.SolveShortestPath(
                    start, goal, rounding, self.solver, self.options))
                if hard_result[-1].is_success():
                    found_solution = True
            if not found_solution:
                print("Second solve failed on all paths.")
                return None, result, hard_result
        else:
            hard_result = [result]
            active_edges = [active_edges[0]]
        return active_edges, result, hard_result

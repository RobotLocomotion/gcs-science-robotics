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

from spp.preprocessing import removeRedundancies
from spp.rounding import (
    MipPathExtraction,
    randomForwardPathSearch,
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
        self.rounding_fn = [randomForwardPathSearch]
        self.rounding_kwargs = {}
        for r in self.regions:
            assert r.ambient_dimension() == self.dimension

        self.spp = GraphOfConvexSets()
        self.graph_complete = True
        self.edge_cost_dict = {}


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

    def setRoundingStrategy(self, rounding_fn, **kwargs):
        self.rounding_kwargs = kwargs
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
        for edge in self.spp.Edges():
            edge.ClearPhiConstraints()
            if edge.u() in vertices or edge.v() in vertices:
                self.edge_cost_dict.pop(edge.id())
        for v in vertices:
            self.spp.RemoveVertex(v)

    def VisualizeGraph(self, file_type="svg"):
        graphviz = self.spp.GetGraphvizString(None, False)
        data = pydot.graph_from_dot_data(graphviz)[0]
        if file_type == "svg":
            return data.create_svg()
        elif file_type == "png":
            return data.create_png()
        else:
            raise ValueError("Unrecognized file type:", file_type)


    def solveSPP(self, start, goal, rounding, preprocessing, verbose):
        if not self.graph_complete:
            raise NotImplementedError(
                "Replanning on a graph that has undergone preprocessing is "
                "not supported yet. Please construct a new planner.")
        statistics = {}
        if preprocessing:
            statistics["preprocessing"] = removeRedundancies(self.spp, start, goal, verbose=verbose)
            self.graph_complete = False

        result = self.spp.SolveShortestPath(start, goal, rounding, self.solver, self.options)

        statistics["solver_time"] = result.get_solver_details().optimizer_time
        statistics["result_cost"] = result.get_optimal_cost()

        if not result.is_success():
            print("First solve failed")
            return None, result, None, statistics

        if verbose:
            print("Solution\t",
                  "Success:", result.get_solution_result(),
                  "Cost:", result.get_optimal_cost(),
                  "Solver time:", result.get_solver_details().optimizer_time)

        # Solve with hard edge choices
        if rounding:
            # Extract path
            active_edges = []
            found_path = False
            for fn in self.rounding_fn:
                rounded_edges = fn(self.spp, result, start, goal,
                                   edge_cost_dict=self.edge_cost_dict,
                                   **self.rounding_kwargs)
                if rounded_edges is None:
                    print(fn.__name__, "could not find a path.")
                    active_edges.append(rounded_edges)
                else:
                    found_path = True
                    active_edges.extend(rounded_edges)
            if not found_path:
                print("All rounding strategies failed to find a path.")
                return None, result, None, statistics

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
                    last_cost = hard_result[-1].get_optimal_cost()
                    if (last_cost - statistics["result_cost"]) / last_cost < .01:
                        break

            statistics["min_hard_solver_time"] =  min(list(map(lambda r: r.get_solver_details().optimizer_time, hard_result)), default = 0.0)
            statistics["min_hard_optimal_cost"] =  min(list(map(lambda r: r.get_optimal_cost(), hard_result)), default = 0.0)

            if verbose:
                print("Rounded Solutions:")
                for r in hard_result:
                    if r is None:
                        print("\t\tNo path to solve")
                        continue
                    print("\t\t",
                        "Success:", r.get_solution_result(),
                        "Cost:", r.get_optimal_cost(),
                        "Solver time:", r.get_solver_details().optimizer_time)

            if not found_solution:
                print("Second solve failed on all paths.")
                return None, result, hard_result, statistics
        else:
            active_edges = MipPathExtraction(self.spp, result, start, goal)
            hard_result = [result]
            statistics["min_hard_solver_time"] =  0.0
        return active_edges, result, hard_result, statistics

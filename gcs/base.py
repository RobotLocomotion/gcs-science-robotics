import pydot
import numpy as np

from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    Point,
)
from pydrake.solvers import (
    CommonSolverOption,
    GurobiSolver,
    MathematicalProgram,
    MosekSolver,
    SolverOptions,
)
from pydrake.all import le

from gcs.rounding import MipPathExtraction

def polytopeDimension(A, b, tol=1e-4):
    
    assert A.shape[0] == b.size
    
    m, n = A.shape
    eq = []

    while True:
        
        ineq = [i for i in range(m) if i not in eq]
        A_ineq = A[ineq]
        b_ineq = b[ineq]

        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(n)
        r = prog.NewContinuousVariables(1)[0]
        
        if len(eq) > 0:
            prog.AddLinearEqualityConstraint(A_eq.dot(x), b_eq)
        if len(ineq) > 0:
            c = prog.AddLinearConstraint(le(A_ineq.dot(x) + r * np.ones(len(ineq)), b_ineq))
        prog.AddBoundingBoxConstraint(0, 1, r)
        
        prog.AddLinearCost(-r)

        solver = MosekSolver()
        result = solver.Solve(prog)

        if not result.is_success():
            return -1
        
        if result.GetSolution(r) > tol:
            eq_rank = 0 if len(eq) == 0 else np.linalg.matrix_rank(A_eq)
            return n - eq_rank
        
        c_opt = np.abs(result.GetDualSolution(c)) 
        eq += [ineq[i] for i, ci in enumerate(c_opt) if ci > tol]
        A_eq = A[eq]
        b_eq = b[eq]

class BaseGCS:
    def __init__(self, regions):
        self.names = None
        if type(regions) is dict:
            self.names = list(regions.keys())
            regions = list(regions.values())
        else:
            self.names = ["v" + str(ii) for ii in range(len(regions))]
        self.dimension = regions[0].ambient_dimension()
        self.regions = regions.copy()
        self.rounding_fn = []
        self.rounding_kwargs = {}
        for r in self.regions:
            assert r.ambient_dimension() == self.dimension

        self.gcs = GraphOfConvexSets()
        self.options = GraphOfConvexSetsOptions()
        self.source = None
        self.target = None

    def addSourceTarget(self, source, target, edges=None):
        if self.source is not None or self.target is not None:
            self.gcs.RemoveVertex(self.source)
            self.gcs.RemoveVertex(self.target)

        assert len(source) == self.dimension
        assert len(target) == self.dimension

        vertices = self.gcs.Vertices()
        # Add edges connecting source and target to graph
        self.source = self.gcs.AddVertex(Point(source), "source")
        self.target = self.gcs.AddVertex(Point(target), "target")

        # Add edges connecting source and target to graph
        if edges is None:
            edges = self.findStartGoalEdges(source, target)

        if not (len(edges[0]) > 0):
            raise ValueError('Source vertex is not connected.')
        if not (len(edges[1]) > 0):
            raise ValueError('Target vertex is not connected.')

        source_edges = []
        target_edges = []
        for ii in edges[0]:
            u = vertices[ii]
            edge = self.gcs.AddEdge(self.source, u, f"(source, {u.name()})")
            source_edges.append(edge)

        for ii in edges[1]:
            u = vertices[ii]
            edge = self.gcs.AddEdge(u, self.target, f"({u.name()}, target)")
            target_edges.append(edge)

        return source_edges, target_edges

    def findEdgesViaOverlaps(self):
        edges = []
        for ii in range(len(self.regions)):
            for jj in range(ii + 1, len(self.regions)):
                if self.regions[ii].IntersectsWith(self.regions[jj]):
                    edges.append((ii, jj))
                    edges.append((jj, ii))
        return edges

    def findEdgesViaFullDimensionOverlaps(self):
        edges = []
        for ii in range(len(self.regions)):
            for jj in range(ii + 1, len(self.regions)):
                A = np.vstack((self.regions[ii].A(), self.regions[jj].A()))
                b = np.concatenate((self.regions[ii].b(), self.regions[jj].b()))
                if polytopeDimension(A, b) >= self.dimension - 1:
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
        self.options.solver = solver

    def setSolverOptions(self, options):
        self.options.solver_options = options

    def setPaperSolverOptions(self):
        solver_options = SolverOptions()
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
        solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", 3600.0)
        solver_options.SetOption(GurobiSolver.id(), "MIPGap", 1e-3)
        solver_options.SetOption(GurobiSolver.id(), "TimeLimit", 3600.0)
        self.options.solver_options = solver_options

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

    def ResetGraph(self, vertices=None):
        if vertices is None:
            vertices = [self.source, self.target]
            self.source = None
            self.target = None
        for v in vertices:
            self.gcs.RemoveVertex(v)
        for edge in self.gcs.Edges():
            edge.ClearPhiConstraints()

    def VisualizeGraph(self, file_type="svg"):
        graphviz = self.gcs.GetGraphvizString(None, False)
        data = pydot.graph_from_dot_data(graphviz)[0]
        if file_type == "svg":
            return data.create_svg()
        elif file_type == "png":
            return data.create_png()
        else:
            raise ValueError("Unrecognized file type:", file_type)


    def solveGCS(self, rounding, preprocessing, verbose):

        results_dict = {}
        self.options.convex_relaxation = rounding
        self.options.preprocessing = preprocessing
        self.options.max_rounded_paths = 0

        result = self.gcs.SolveShortestPath(self.source, self.target, self.options)

        if rounding:
            results_dict["relaxation_result"] = result
            results_dict["relaxation_solver_time"] = result.get_solver_details().optimizer_time
            results_dict["relaxation_cost"] = result.get_optimal_cost()
        else:
            results_dict["mip_result"] = result
            results_dict["mip_solver_time"] = result.get_solver_details().optimizer_time
            results_dict["mip_cost"] = result.get_optimal_cost()

        if not result.is_success():
            print("First solve failed")
            return None, None, results_dict

        if verbose:
            print("Solution\t",
                  "Success:", result.get_solution_result(),
                  "Cost:", result.get_optimal_cost(),
                  "Solver time:", result.get_solver_details().optimizer_time)

        # Solve with hard edge choices
        if rounding and len(self.rounding_fn) > 0:
            # Extract path
            active_edges = []
            found_path = False
            for fn in self.rounding_fn:
                rounded_edges = fn(self.gcs, result, self.source, self.target,
                                   **self.rounding_kwargs)
                if rounded_edges is None:
                    print(fn.__name__, "could not find a path.")
                    active_edges.append(rounded_edges)
                else:
                    found_path = True
                    active_edges.extend(rounded_edges)
            results_dict["rounded_paths"] = active_edges
            if not found_path:
                print("All rounding strategies failed to find a path.")
                return None, None, results_dict

            self.options.preprocessing = False
            rounded_results = []
            best_cost = np.inf
            best_path = None
            best_result = None
            max_rounded_solver_time = 0.0
            total_rounded_solver_time = 0.0
            for path_edges in active_edges:
                if path_edges is None:
                    rounded_results.append(None)
                    continue
                for edge in self.gcs.Edges():
                    if edge in path_edges:
                        edge.AddPhiConstraint(True)
                    else:
                        edge.AddPhiConstraint(False)
                rounded_results.append(self.gcs.SolveShortestPath(
                    self.source, self.target, self.options))
                solve_time = rounded_results[-1].get_solver_details().optimizer_time
                max_rounded_solver_time = np.maximum(solve_time, max_rounded_solver_time)
                total_rounded_solver_time += solve_time
                if (rounded_results[-1].is_success()
                    and rounded_results[-1].get_optimal_cost() < best_cost):
                    best_cost = rounded_results[-1].get_optimal_cost()
                    best_path = path_edges
                    best_result = rounded_results[-1]

            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["rounded_results"] = rounded_results
            results_dict["max_rounded_solver_time"] =  max_rounded_solver_time
            results_dict["total_rounded_solver_time"] = total_rounded_solver_time
            results_dict["rounded_cost"] = best_result.get_optimal_cost()

            if verbose:
                print("Rounded Solutions:")
                for r in rounded_results:
                    if r is None:
                        print("\t\tNo path to solve")
                        continue
                    print("\t\t",
                        "Success:", r.get_solution_result(),
                        "Cost:", r.get_optimal_cost(),
                        "Solver time:", r.get_solver_details().optimizer_time)

            if best_path is None:
                print("Second solve failed on all paths.")
                return best_path, best_result, results_dict
        elif rounding:
            self.options.max_rounded_paths = 10

            rounded_result = self.gcs.SolveShortestPath(self.source, self.target, self.options)
            best_path = MipPathExtraction(self.gcs, rounded_result, self.source, self.target)[0]
            best_result = rounded_result
            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["rounded_results"] = [rounded_result]
            results_dict["rounded_cost"] = best_result.get_optimal_cost()
        else:
            best_path = MipPathExtraction(self.gcs, result, self.source, self.target)[0]
            best_result = result
            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["mip_path"] = best_path

        if verbose:
            for edge in best_path:
                print("Added", edge.name(), "to path.")

        return best_path, best_result, results_dict

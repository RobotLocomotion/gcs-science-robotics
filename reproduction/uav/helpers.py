import numpy as np
import os
import pickle
import time

from pydrake.systems.framework import LeafSystem
from pydrake.solvers import MosekSolver

from gcs.bezier import BezierGCS
from gcs.rounding import *
from reproduction.uav.building_generation import *

def generate_buildings(save_location, num_buildings):
    start = np.array([-1, -1])
    goal = np.array([2, 1])
    building_shape = (3, 3)

    for ii in range(num_buildings):
        file_location = save_location + "/room_" + str(ii).zfill(3)

        if not os.path.exists(file_location):
            os.makedirs(file_location)

        grid, outdoor_edges, wall_edges = generate_grid_world(shape=building_shape, start=start, goal=goal)
        regions = compile_sdf(file_location + "/building.sdf", grid, start, goal, outdoor_edges, wall_edges)
        with open(file_location + '/regions.reg', 'wb') as f:
            pickle.dump(regions, f)

def build_bezier_gcs(regions, solver):
    order = 7
    continuity = 4
    vel_limit = 10 * np.ones(3)
    hdot_min = 1e-3
    weights = {"time": 1., "norm": 1.}
    max_paths = 10 # default
    max_trials = 100 # default
    rounding_seed = 0

    gcs = BezierGCS(regions, order, continuity, hdot_min=hdot_min, full_dim_overlap=True)
    gcs.addTimeCost(weights["time"])
    gcs.addPathLengthCost(weights["norm"])
    gcs.addVelocityLimits(-vel_limit, vel_limit)
    gcs.setPaperSolverOptions()
    gcs.setSolver(solver)
    gcs.setRoundingStrategy(randomForwardPathSearch, max_paths=max_paths, max_trials=max_trials, seed=rounding_seed)
#     regularization = 1e-3
#     for derivative in [2, 3, 4]:
#         gcs.addDerivativeRegularization(regularization, regularization, derivative)

    return gcs

def plan_through_buildings(save_location, num_buildings, solve_gcs=True, file_addendum=None, solver=None):
    start = np.array([-1, -1])
    goal = np.array([2, 1])
    start_pose = np.r_[(start-start)*5, 1.]
    goal_pose = np.r_[(goal-start)*5., 1.]

    if solver is None:
        solver = MosekSolver()

    for ii in range(num_buildings):
        file_location = save_location + "/room_" + str(ii).zfill(3)

        with open(file_location + "/regions.reg", "rb") as f:
            regions = pickle.load(f)

        start_setup = time.time()
        gcs = build_bezier_gcs(regions, solver)
        gcs.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
        setup_time = time.time() - start_setup

        planning_results = dict()
        planning_results["order"] = gcs.order
        planning_results["continuity"] = gcs.continuity
        planning_results["start_pose"] = start_pose
        planning_results["goal_pose"] = goal_pose
        planning_results["setup_time"] = setup_time

        start_time = time.time()
        output = gcs.SolvePath(solve_gcs, False, preprocessing=True)
        b_traj, results_dict = output
        solve_time = time.time() - start_time

        print("Solve time for building", ii, ":", solve_time, flush=True)

        planning_results["preprocessing_time"] = results_dict["preprocessing_stats"]["linear_programs"]

        if solve_gcs:
            planning_results["gcs_time"] = solve_time

            result = results_dict["relaxation_result"]
            planning_results["relaxation_result"] = result.get_solution_result()
            planning_results["relaxation_time"] = results_dict["relaxation_solver_time"]
            planning_results["relaxation_cost"] = results_dict["relaxation_cost"]
            planning_results["relaxation_solution"] = []
            for edge in gcs.gcs.Edges():
                edge_solution = {"name": edge.name(),
                                 "y_e": edge.GetSolutionPhiXu(result),
                                 "z_e": edge.GetSolutionPhiXv(result),
                                 "phi_e": result.GetSolution(edge.phi())}
                planning_results["relaxation_solution"].append(edge_solution)

            best_result = results_dict["best_result"]
            if best_result is not None:
                planning_results["rounded_result"] = best_result.get_solution_result()
                planning_results["rounded_time"] = results_dict["total_rounded_solver_time"]
                planning_results["gcs_solver_time"] = (planning_results["relaxation_time"]
                                                       + planning_results["rounded_time"]
                                                       + planning_results["preprocessing_time"])
                planning_results["rounded_cost"] = results_dict["rounded_cost"]
                planning_results["rounded_solution"] = []
                for edge in gcs.gcs.Edges():
                    edge_solution = {"name": edge.name(),
                                     "y_e": edge.GetSolutionPhiXu(best_result),
                                     "z_e": edge.GetSolutionPhiXv(best_result),
                                     "phi_e": best_result.GetSolution(edge.phi())}
                    planning_results["rounded_solution"].append(edge_solution)

            else:
                planning_results["rounded_result"] = None
                planning_results["rounded_time"] = np.nan
                planning_results["gcs_solver_time"] = np.nan
                planning_results["rounded_cost"] = np.nan
                planning_results["rounded_solution"] = None

            traj_file = file_location + "/relaxation_traj.pkl"
            result_file = file_location + "/relaxation_plan_results.pkl"
            if file_addendum is not None:
                traj_file = file_location + "/relaxation_traj_" + file_addendum + ".pkl"
                result_file = file_location + "/relaxation_plan_results_" + file_addendum + ".pkl"
            with open(traj_file, "wb") as f:
                pickle.dump(b_traj, f, pickle.HIGHEST_PROTOCOL)
            with open(result_file, 'wb') as f:
                pickle.dump(planning_results, f)

        else:
            planning_results["mip_time"] = solve_time

            result = results_dict["mip_result"]
            planning_results["mip_result"] = result.get_solution_result()
            planning_results["mip_solver_time"] = results_dict["mip_solver_time"]
            planning_results["mip_total_solver_time"] = (planning_results["preprocessing_time"]
                                                         + planning_results["mip_solver_time"])
            planning_results["mip_cost"] = results_dict["mip_cost"]
            planning_results["mip_solution"] = []
            for edge in gcs.gcs.Edges():
                edge_solution = {"name": edge.name(),
                                 "y_e": edge.GetSolutionPhiXu(result),
                                 "z_e": edge.GetSolutionPhiXv(result),
                                 "phi_e": result.GetSolution(edge.phi())}
                planning_results["mip_solution"].append(edge_solution)

            traj_file = file_location + "/mip_traj.pkl"
            result_file = file_location + "/mip_plan_results.pkl"
            if file_addendum is not None:
                traj_file = file_location + "/mip_traj_" + file_addendum + ".pkl"
                result_file = file_location + "/mip_plan_results_" + file_addendum + ".pkl"
            with open(traj_file, "wb") as f:
                pickle.dump(b_traj, f, pickle.HIGHEST_PROTOCOL)
            with open(result_file, 'wb') as f:
                pickle.dump(planning_results, f)

class FlatnessInverter(LeafSystem):
    def __init__(self, traj, animator, t_offset=0):
        LeafSystem.__init__(self)
        self.traj = traj
        self.port = self.DeclareVectorOutputPort("state", 12, self.DoCalcState, {self.time_ticket()})
        self.t_offset = t_offset
        self.animator = animator

    def DoCalcState(self, context, output):
        t = context.get_time() + self.t_offset - 1e-4

        q = np.squeeze(self.traj.value(t))
        q_dot = np.squeeze(self.traj.EvalDerivative(t))
        q_ddot = np.squeeze(self.traj.EvalDerivative(t, 2))

        fz = np.sqrt(q_ddot[0]**2 + q_ddot[1]**2 + (q_ddot[2] + 9.81)**2)
        r = np.arcsin(-q_ddot[1]/fz)
        p = np.arcsin(q_ddot[0]/fz)

        output.set_value(np.concatenate((q, [r, p, 0], q_dot, np.zeros(3))))

        if self.animator is not None:
            frame = self.animator.frame(context.get_time())
            self.animator.SetProperty(frame, "/Cameras/default/rotated/<object>", "position", [-2.5, 4, 2.5])
            self.animator.SetTransform(frame, "/drake", RigidTransform(-q))

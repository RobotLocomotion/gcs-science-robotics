from pydrake.planning.common_robotics_utilities import (
    Graph, GraphNode,
    GrowRoadMap,
    UpdateRoadMapEdges,
    QueryPath,
    LazyQueryPath,
    AddNodeToRoadmap,
    NNDistanceDirection,
    ShortcutSmoothPath,
    SimpleRRTPlannerState,
    MakeKinematicLinearBiRRTNearestNeighborsFunction,
    MakeBiRRTTimeoutTerminationFunction, 
    BiRRTPlanSinglePath, PropagatedState)
from pydrake.common import RandomGenerator
import numpy as np
import pickle
import time

class ClassicalPlanning:
    def __init__(self, plant, plant_context, step_size, collision_step_size):
        self.plant = plant
        self.plant_context = plant_context
        self.step_size = step_size
        self.collision_step_size = collision_step_size

        self.PositionUpperLimits = self.plant.GetPositionUpperLimits()
        self.PositionLowerLimits = self.plant.GetPositionLowerLimits()

    def distance_fn(self, q_1, q_2):
        return np.linalg.norm(q_2 - q_1)

    def check_state_validity_fn(self, q):
        self.plant.SetPositions(self.plant_context, q)
        query_object = self.plant.get_geometry_query_input_port().Eval(self.plant_context)
    
        return not query_object.HasCollisions()       

    def InterpolateWaypoint(self, start, end, ratio):
        return start + ratio*(end-start)

    def check_edge_validity_fn(self, start, end):
        num_steps = np.ceil(self.distance_fn(start, end)/self.collision_step_size)

        for step in range(int(num_steps)+1):
            interpolation_ratio = step / num_steps
            interpolated_point = self.InterpolateWaypoint(start, end, interpolation_ratio)
            
            if not self.check_state_validity_fn(interpolated_point):
                return False

        return True
    
    def shortcut(self, path, max_iter, max_failed_iter, max_backtracking_steps,max_shortcut_fraction,
                 resample_shortcuts_interval, check_for_marginal_shortcuts, seed = 0):

        shortcutted_path = ShortcutSmoothPath(path,
                                              max_iter,
                                              max_failed_iter,
                                              max_backtracking_steps,
                                              max_shortcut_fraction,
                                              resample_shortcuts_interval,
                                              check_for_marginal_shortcuts,
                                              self.check_edge_validity_fn, 
                                              self.distance_fn,
                                              self.InterpolateWaypoint,
                                              RandomGenerator(seed))
        return shortcutted_path
 



class PRM(ClassicalPlanning):
    def __init__(self, plant, plant_context, step_size, collision_step_size, K, solve_timeout):
        ClassicalPlanning.__init__(self, plant, plant_context, step_size, collision_step_size)
        self.K = K
        self.solve_timeout = solve_timeout
        self.roadmap = Graph()
        self.roadmap_size = 0

    def check_goal_fn(self, q):
        return np.linalg.norm(q_end - q) < 1e-6

    def prm_sampling_fn(self):
        return np.random.rand(len(self.PositionLowerLimits))*(self.PositionUpperLimits-self.PositionLowerLimits) + self.PositionLowerLimits

    def roadmap_termination_fn(self, current_roadmap_size):
        return current_roadmap_size >= self.roadmap_size
    
    def GrowRoadMap(self, target_size):
        if target_size <= 0:
            raise Exception(f"Target size must be greater then 0")
        if  target_size <= self.roadmap_size:
            raise Exception(f"Target size of {target_size} must be greater then previous size {self.roadmap_size}")

        self.roadmap_size = target_size
        data_stats = GrowRoadMap(self.roadmap, self.prm_sampling_fn, self.distance_fn,
                                self.check_state_validity_fn, self.check_edge_validity_fn,
                                self.roadmap_termination_fn, self.K, False, True, False)
        return data_stats
    
    def resetRoadmap(self):
        self.roadmap = Graph()
        self.roadmap_size = 0
    
    def addNodes(self, q_list):
        for q in q_list:
            AddNodeToRoadmap(q, NNDistanceDirection(), self.roadmap, self.distance_fn, self.check_edge_validity_fn, self.K, False,True,False )

        self.roadmap_size = self.roadmap.Size()


    def getPath(self, sequence, verbose = False, path_processing = lambda path: path):
        path = [sequence[0]]
        run_time = 0.0
        for start_pt, goal_pt in zip(sequence[:-1], sequence[1:]):
            start_time = time.time()
            prm_path = QueryPath([start_pt], [goal_pt], self.roadmap, self.distance_fn,
                    self.check_edge_validity_fn, self.K,
                    use_parallel=False,
                    distance_is_symmetric=True,
                    add_duplicate_states=False,
                    limit_astar_pqueue_duplicates=True).Path()
            
            prm_path = path_processing(prm_path)
            run_time += time.time() - start_time

            if len(prm_path) == 0:
                if verbose:
                    print(f"Failed between {start_pt} and {goal_pt}")
                return None
            
            path += prm_path[1:]
                
        return np.stack(path).T, run_time

    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.roadmap,f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.roadmap = pickle.load(f)
            self.roadmap_size = self.roadmap.Size()


class BiRRT(ClassicalPlanning):
    def __init__(self, plant, plant_context, step_size, collision_step_size, goal_bias, solve_timeout):
        ClassicalPlanning.__init__(self, plant, plant_context, step_size, collision_step_size)
        self.goal_bias = goal_bias
        self.solve_timeout = solve_timeout
    
    def connect_fn(self, nearest, sample, is_start_tree):
        total_dist = self.distance_fn(nearest, sample)
        total_steps = int(np.ceil(total_dist / self.step_size))        
            
        propagated_states = []
        parent_offset = -1
        current = nearest
        for steps in range(total_steps):
            current_target = None
            target_dist = self.distance_fn(current, sample)
            if target_dist > self.step_size:
                #interpolate
                current_target = self.InterpolateWaypoint(current, sample, self.step_size/target_dist)
                
            elif target_dist < 1e-6:
                break
            else:
                current_target = sample
            
            if not self.check_edge_validity_fn(current, current_target):
                return propagated_states
        
                        
            propagated_states.append(PropagatedState(state=current_target, relative_parent_index=parent_offset))
            parent_offset += 1
            current = current_target

        return propagated_states

    def states_connected_fn(self, source, target, is_start_tree):
        return np.linalg.norm(source - target) < 1e-6
    

    def RRT_Connect(self, q_ini, q_final, seed = 0):
        start_tree = [SimpleRRTPlannerState(q_ini)]
        end_tree = [SimpleRRTPlannerState(q_final)]
        
        def birrt_sampling():
            if np.random.rand() < self.goal_bias:
                if np.random.rand() < 0.5:
                    return q_ini
                else:
                    return q_final
            return np.random.rand(len(self.PositionLowerLimits))*(self.PositionUpperLimits - self.PositionLowerLimits) + self.PositionLowerLimits

        nearest_neighbor_fn = MakeKinematicLinearBiRRTNearestNeighborsFunction(distance_fn = self.distance_fn, use_parallel = False)

        termination_fn = MakeBiRRTTimeoutTerminationFunction(self.solve_timeout)
        
        connect_result = BiRRTPlanSinglePath(
                start_tree=start_tree, goal_tree=end_tree,
                state_sampling_fn=birrt_sampling,
                nearest_neighbor_fn=nearest_neighbor_fn, propagation_fn=self.connect_fn,
                state_added_callback_fn=None,
                states_connected_fn=self.states_connected_fn,
                goal_bridge_callback_fn=None,
                tree_sampling_bias=0.5, p_switch_tree=0.25,
                termination_check_fn=termination_fn, rng=RandomGenerator(seed))
        return connect_result 

    def getPath(self, sequence, verbose = False, path_processing = lambda path: path):
        path = [sequence[0]]
        run_time = 0.0
        for start_pt, goal_pt in zip(sequence[:-1], sequence[1:]):

            start_time = time.time()
            path = self.RRT_Connect(start_pt, goal_pt).Path()

            path = path_processing(path)
            run_time += time.time() - start_time

            if len(path) == 0:
                if verbose:
                    print(f"Failed between {start_pt} and {goal_pt}")
                return None
            
            path += path[1:]
            
        return np.stack(path).T, run_time


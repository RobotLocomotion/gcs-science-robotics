from typing import Tuple, Sequence, Optional
from pydrake.all import (PRMPlanner, PathProcessor,
                         HolonomicKinematicPlanningSpace, JointLimits,
                         VoxelizedEnvironmentCollisionChecker,
                         RobotDiagramBuilder, LoadModelDirectives,
                         ProcessModelDirectives, PRMPlannerQueryParameters,
                         PathProcessorParameters)

from reproduction.util import FindModelFile, GcsDir
import pickle
import numpy as np


class PresplinedPRM:

    def __init__(
        self,
        edge_step_size: float = 0.05,
        env_padding: float = 0.01,
        self_padding: float = 0.01,
        propagation_step_size: float = 0.5,
        grid_size: Tuple[float] = (2.0, 2.0, 2.0),
        grid_resolution: float = 0.02,
        seed: int = 0,
    ):
        """Initialize the planner.

        Args:
            edge_step_size: The step size to use when checking for collisions
            between two configurations.
            env_padding: The padding to use when voxelizing the environment.
            propagation_step_size: The step size to use when propagating
                between two configurations.
            grid_size: The size of the voxelized environment.
            grid_resolution: The size of each voxel.
            seed: The seed to use for the random number generator.
        """

        # Build the scene.
        builder = RobotDiagramBuilder(time_step=0.0)
        builder.parser().package_map().Add("gcs", GcsDir())

        directives_file = FindModelFile(
            "models/iiwa14_spheres_collision_welded_gripper.yaml")
        directives = LoadModelDirectives(directives_file)
        ProcessModelDirectives(directives, builder.parser())

        iiwa_idx = builder.plant().GetModelInstanceByName("iiwa")
        wsg_idx = builder.plant().GetModelInstanceByName("wsg")

        named_joint_distance_weights = dict()
        named_joint_distance_weights["iiwa_joint_1"] = 1.0
        named_joint_distance_weights["iiwa_joint_2"] = 1.0
        named_joint_distance_weights["iiwa_joint_3"] = 1.0
        named_joint_distance_weights["iiwa_joint_4"] = 1.0
        named_joint_distance_weights["iiwa_joint_5"] = 1.0
        named_joint_distance_weights["iiwa_joint_6"] = 1.0
        named_joint_distance_weights["iiwa_joint_7"] = 1.0

        builder.plant().Finalize()
        joint_limits = JointLimits(builder.plant())
        diagram = builder.Build()

        robot_model_instances = [iiwa_idx, wsg_idx]

        # Build collision checker.
        collision_checker = VoxelizedEnvironmentCollisionChecker(
            model=diagram,
            robot_model_instances=robot_model_instances,
            edge_step_size=edge_step_size,
            env_collision_padding=env_padding,
            self_collision_padding=self_padding,
            named_joint_distance_weights=named_joint_distance_weights,
            default_joint_distance_weight=1.0,
        )

        collision_checker.VoxelizeEnvironment(grid_size, grid_resolution)

        # Make the planning space.
        self.planning_space = HolonomicKinematicPlanningSpace(
            collision_checker, joint_limits, propagation_step_size, seed)
        self.roadmap = None

    def plan(
        self,
        sequence: Sequence[np.array],
        query_paramters: PRMPlannerQueryParameters,
        post_processing_parameters: Optional[PathProcessorParameters] = None
    ) -> Tuple[np.array, float]:
        """Plan a path through the sequence of configurations.

        Args:
            sequence: The sequence of configurations to plan through.
            prm_query_parameters: The parameters to use for the PRM.
            post_processing_parameters: The parameters to use for post
                processing. If None, no post processing is done.

            Returns:
                The path through the sequence of configurations and the
                total run time.
        """
        if self.roadmap is None:
            raise Exception("Roadmap not built yet.")
        path = [sequence[0]]
        run_time = 0.0
        for start, goal in zip(sequence[:-1], sequence[1:]):
            path_result, prm_run_time = PRMPlanner.TimedPlanLazy(
                start, goal, query_paramters, self.planning_space,
                self.roadmap)
            run_time += prm_run_time

            if not path_result.has_solution():
                print(f"Failed between {start} and {goal}")
                return None, run_time

            if post_processing_parameters is not None:
                processed_path, processing_time = PathProcessor.TimedProcessPath(
                    path_result.path(), post_processing_parameters,
                    self.planning_space)
                run_time += processing_time
                path += processed_path[1:]
            else:
                path += path_result.path()[1:]

        return np.stack(path).T, run_time

    def save(self, filename) -> None:
        """Save the roadmap to a file.

        Args:
            filename: The name of the file to save the roadmap to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.roadmap, f)

    def load(self, filename) -> None:
        """Load the roadmap from a file.

        Args:
            filename: The name of the file to load the roadmap from.
        """
        with open(filename, 'rb') as f:
            self.roadmap = pickle.load(f)

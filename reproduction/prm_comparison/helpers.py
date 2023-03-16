from typing import Sequence, List, Union, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import (IllustrationProperties, MeshcatVisualizer,
                              MeshcatVisualizerParams, Rgba, RoleAssign, Role,
                              SceneGraph)
from pydrake.geometry.optimization import VPolytope, HPolyhedron
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.perception import PointCloud
from pydrake.all import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import TrajectorySource, Multiplexer, ConstantVectorSource
from pydrake.multibody.tree import RevoluteJoint
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.analysis import Simulator
from pydrake.multibody import inverse_kinematics
from pydrake.solvers import Solve

from reproduction.util import GcsDir, FindModelFile


def ForwardKinematics(q_list: List[Sequence[float]]) -> List[RigidTransform]:
    """Returns the end-effector pose for the given joint angles.

    The end-effector is the body of the wsg gripper.

    Args:
        q_list: List of joint angles.

    Returns:
        List of end-effector poses.
    """
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().Add("gcs", GcsDir())

    directives_file = FindModelFile(
        "models/iiwa14_spheres_collision_welded_gripper.yaml")
    directives = LoadModelDirectives(directives_file)
    ProcessModelDirectives(directives, plant, parser)

    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    X_list = []
    for q in q_list:
        plant.SetPositions(plant_context, q)
        X_list.append(
            plant.EvalBodyPoseInWorld(plant_context,
                                      plant.GetBodyByName("body")))

    return X_list


def InverseKinematics(q0: Sequence, translation: Sequence,
                      rpy: Sequence) -> Sequence:
    """Returns the joint angles for the given end-effector pose.

    Args:
        q0: Initial guess for the joint angles.
        translation: Desired end-effector translation.
        rpy: Desired end-effector orientation in roll-pitch-yaw.

    Returns:
        The joint angles for the given end-effector pose.
    """
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().Add("gcs", GcsDir())

    directives_file = FindModelFile(
        "models/iiwa14_spheres_collision_welded_gripper.yaml")
    directives = LoadModelDirectives(directives_file)
    ProcessModelDirectives(directives, plant, parser)

    plant.Finalize()

    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    gripper_frame = plant.GetBodyByName("body").body_frame()
    ik = inverse_kinematics.InverseKinematics(plant, plant_context)
    ik.AddPositionConstraint(gripper_frame, [0, 0, 0], plant.world_frame(),
                             translation, translation)

    ik.AddOrientationConstraint(gripper_frame, RotationMatrix(),
                                plant.world_frame(),
                                RotationMatrix(RollPitchYaw(*rpy)), 0.001)

    prog = ik.get_mutable_prog()
    q = ik.q()

    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0)
    result = Solve(ik.prog())
    if not result.is_success():
        print("IK failed")
        return None
    q1 = result.GetSolution(q)
    return q1


def set_transparency_of_models(plant, model_instances, alpha, scene_graph):
    """Sets the transparency of the given models."""
    inspector = scene_graph.model_inspector()
    for model in model_instances:
        for body_id in plant.GetBodyIndices(model):
            frame_id = plant.GetBodyFrameIdOrThrow(body_id)
            for geometry_id in inspector.GetGeometries(frame_id,
                                                       Role.kIllustration):
                properties = inspector.GetIllustrationProperties(geometry_id)
                phong = properties.GetProperty("phong", "diffuse")
                phong.set(phong.r(), phong.g(), phong.b(), alpha)
                properties.UpdateProperty("phong", "diffuse", phong)
                scene_graph.AssignRole(plant.get_source_id(), geometry_id,
                                       properties, RoleAssign.kReplace)


def combine_trajectories(
        traj_list: List[PiecewisePolynomial.FirstOrderHold],
        wait: float = 2.0) -> PiecewisePolynomial.FirstOrderHold:
    """Combines multiple trajectories into one.

    Args:
        traj_list: List of trajectories to be combined.
        wait: Wait time between trajectories (seconds).

    Returns:
        A combined trajectory.
    """
    knotList = []
    time_list = []
    for traj in traj_list:
        knots = traj.vector_values(traj.get_segment_times()).T
        knotList.append(knots)

        duration = traj.end_time() - traj.start_time()
        offset = time_list[-1][-1] + 0.1 if time_list else 0
        time_list.append(np.linspace(offset, duration + offset,
                                     knots.shape[0]))

        # Add wait time.
        if wait > 0.0:
            knotList.append(knotList[-1][-1, :])
            time_list.append(np.array([time_list[-1][-1] + wait]))

    path = np.vstack(knotList).T
    time_break = np.hstack(time_list)

    return PiecewisePolynomial.FirstOrderHold(time_break, path)


def make_traj(path: np.array,
              speed: float = 2) -> PiecewisePolynomial.FirstOrderHold:
    """Returns a trajectory for the given path with constant speed.

    Args:
        path: The path to be turned into a trajectory.
        speed: The speed of the trajectory.
    Returns:
        A trajectory for the given path with constant speed.
    """
    t_breaks = [0]
    distance_between_knots = np.sqrt(
        np.sum(np.square(path.T[1:, :] - path.T[:-1, :]), axis=1))
    for segment_duration in distance_between_knots / speed:
        # Add a small number to ensure all times are increasing.
        t_breaks += [segment_duration + 1e-6 + t_breaks[-1]]
    return PiecewisePolynomial.FirstOrderHold(t_breaks, path)


def get_traj_length(trajectory: PiecewisePolynomial.FirstOrderHold,
                    weights: Union[float, np.array] = 1) -> float:
    """Returns the length of the trajectory.

    Args:
        trajectory: The trajectory to be evaluated.
        weights: The weights for the different joint angles. Default is 1 for all joints.

    Returns:
        The length of the trajectory.
    """
    path_length = 0
    knots = trajectory.vector_values(trajectory.get_segment_times())

    for i in range(knots.shape[1] - 1):
        path_length += np.linalg.norm(
            (knots[:, i + 1] - knots[:, i]).dot(weights))

    return path_length


def visualize_trajectory(meshcat,
                         trajectories: List[
                             PiecewisePolynomial.FirstOrderHold],
                         show_path: bool = False,
                         robot_configurations: Optional[np.array] = None,
                         transparency: float = 1.0,
                         regions: Optional[List[HPolyhedron]] = None) -> None:
    """Visualizes the given trajectories in meshcat.

    Args:
        meshcat: The meshcat visualizer.
        trajectories: A list of trajectories to be executed with a two second
            delay. Supports up to three trajectories.
        show_path: If True, the path of the end-effector base frame is shown
            for each trajectory in a different color.
        robot_configurations: A list of robot configurations to be visualized
            statically.
        transparency: The transparency of the robot configurations. Values
            between 0 and 1.
        regions: A list of regions to be visualized. Each region will be
            translated to a vpolytope, where the vertices will be used to
            visualize the corresponding robot configurations end-effector
            position as a point in task space.
    """

    if len(trajectories) > 3:
        raise ValueError("Only up to three trajectories are supported.")
    combined_traj = combine_trajectories(trajectories)

    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    plant = MultibodyPlant(time_step=0.0)
    plant.RegisterAsSourceForSceneGraph(scene_graph)

    parser = Parser(plant, scene_graph)
    parser.package_map().Add("gcs", GcsDir())

    directives_file = FindModelFile(
        "models/iiwa14_spheres_collision_welded_gripper.yaml")
    directives = LoadModelDirectives(directives_file)
    iiwa, wsg, _, _, _, _ = ProcessModelDirectives(directives, plant, parser)

    # Set transparency of main arm and gripper.
    set_transparency_of_models(plant,
                               [iiwa.model_instance, wsg.model_instance],
                               transparency, scene_graph)

    # Add static configurations of the iiwa for visalization.
    if robot_configurations is not None:
        iiwa_file = FindResourceOrThrow(
            "drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_collision.urdf"
        )
        wsg_file = FindModelFile("models/schunk_wsg_50_welded_fingers.sdf")

        for i, q in enumerate(robot_configurations):
            # Add iiwa and wsg for visualization.
            new_iiwa = parser.AddModelFromFile(iiwa_file, f"vis_iiwa_{i}")
            new_wsg = parser.AddModelFromFile(wsg_file, f"vis_wsg_{i}")

            # Weld iiwa to the world frame.
            plant.WeldFrames(plant.world_frame(),
                             plant.GetFrameByName("base", new_iiwa),
                             RigidTransform())
            # Weld wsg to the iiwa end-effector.
            plant.WeldFrames(
                plant.GetFrameByName("iiwa_link_7", new_iiwa),
                plant.GetFrameByName("body", new_wsg),
                RigidTransform(rpy=RollPitchYaw([np.pi / 2., 0, 0]),
                               p=[0, 0, 0.114]))

            set_transparency_of_models(plant, [new_iiwa, new_wsg],
                                       transparency, scene_graph)
    plant.Finalize()
    # Set default joint angles.
    if robot_configurations:
        plant.SetDefaultPositions(
            np.hstack([np.zeros(7)] + robot_configurations))
    else:
        plant.SetDefaultPositions(np.zeros(7))

    # Add the trajectory source to the diagram.
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(to_pose.get_output_port(),
                    scene_graph.get_source_pose_port(plant.get_source_id()))

    traj_system = builder.AddSystem(TrajectorySource(combined_traj))

    mux = builder.AddSystem(
        Multiplexer([7 for _ in range(1 + len(robot_configurations))]))
    builder.Connect(traj_system.get_output_port(), mux.get_input_port(0))

    if robot_configurations is not None:
        for i, q in enumerate(robot_configurations):
            ghost_pos = builder.AddSystem(ConstantVectorSource(q))
            builder.Connect(ghost_pos.get_output_port(),
                            mux.get_input_port(1 + i))

    builder.Connect(mux.get_output_port(), to_pose.get_input_port())

    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.delete_on_initialization_event = False
    meshcat_params.role = Role.kIllustration
    visalizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat,
                                               meshcat_params)
    meshcat.Delete()

    diagram = builder.Build()

    if show_path:
        X_lists = []
        for traj in trajectories:
            X_list = ForwardKinematics(
                traj.vector_values(
                    np.linspace(traj.start_time(), traj.end_time(),
                                15000)).T.tolist())
            X_lists.append(X_list)

        c_list_rgb = [[i / 255 for i in (0, 0, 255, 255)],
                      [i / 255 for i in (255, 191, 0, 255)],
                      [i / 255 for i in (255, 64, 0, 255)]]

        for i, X_list in enumerate(X_lists):
            pointcloud = PointCloud(len(X_list))
            pointcloud.mutable_xyzs()[:] = np.array(
                list(map(lambda X: X.translation(), X_list))).T[:]
            meshcat.SetObject("paths/" + str(i),
                              pointcloud,
                              0.015 + i * 0.005,
                              rgba=Rgba(*c_list_rgb[i]))

    if regions is not None:
        reg_colors = plt.cm.viridis(np.linspace(0, 1, len(regions)))
        reg_colors[:, 3] = 1.0

        for i, reg in enumerate(regions):
            X_reg = ForwardKinematics(VPolytope(reg).vertices().T)
            pointcloud = PointCloud(len(X_reg))
            pointcloud.mutable_xyzs()[:] = np.array(
                list(map(lambda X: X.translation(), X_reg))).T[:]
            meshcat.SetObject("regions/" + str(i),
                              pointcloud,
                              0.015,
                              rgba=Rgba(*reg_colors[i]))

    simulator = Simulator(diagram)
    visalizer.StartRecording()
    simulator.AdvanceTo(combined_traj.end_time())
    visalizer.PublishRecording()


def make_result_table(gcs_data: dict, prm_data: dict,
                      sprm_data: dict) -> pd.DataFrame:
    """Returns a table with the results of the given data."""
    gcs_len = np.mean(
        [gcs_data[k]['Path Length (rad)'] for k in gcs_data.keys()], axis=1)
    prm_len = np.mean(
        [prm_data[k]['Path Length (rad)'] for k in prm_data.keys()], axis=1)
    sprm_len = np.mean(
        [sprm_data[k]['Path Length (rad)'] for k in sprm_data.keys()], axis=1)

    gcs_time = np.mean([gcs_data[k]['Time (ms)'] for k in gcs_data.keys()],
                       axis=1)
    prm_time = np.mean([prm_data[k]['Time (ms)'] for k in prm_data.keys()],
                       axis=1)
    sprm_time = np.mean([sprm_data[k]['Time (ms)'] for k in sprm_data.keys()],
                        axis=1)
    cols = {
        "GCS Planner": sum(zip(gcs_len, gcs_time), ()),
        "Regular PRM": sum(zip(prm_len, prm_time), ()),
        "Shortcut PRM": sum(zip(sprm_len, sprm_time), ())
    }

    index = pd.MultiIndex.from_tuples(sum(
        zip([(task_name, 'length (rad)') for task_name in gcs_data.keys()],
            [(task_name, 'runtime (ms)') for task_name in gcs_data.keys()]),
        ()),
                                      names=["Task", ""])

    df = pd.DataFrame(data=cols, index=index)

    return df


def plot_results(gcs_data: dict, prm_data: dict, sprm_data: dict) -> None:
    """Plots the results of the given data."""
    plt.figure(figsize=(4.5, 7))

    gcs = [gcs_data[k]['Path Length (rad)'] for k in gcs_data.keys()]
    prm = [prm_data[k]['Path Length (rad)'] for k in prm_data.keys()]
    sprm = [sprm_data[k]['Path Length (rad)'] for k in sprm_data.keys()]

    plt.subplot(2, 1, 1)
    _make_bars(gcs, prm, sprm)
    plt.ylabel('Trajectory length (rad)')
    plt.gca().set_xticklabels([])
    plt.ylim([0, max([np.max(gcs), np.max(prm), np.max(sprm)]) * 1.3])
    plt.legend(loc='upper center',
               bbox_to_anchor=(0.5, 1.15),
               ncol=3,
               fancybox=True)

    gcs = [gcs_data[k]['Time (ms)'] for k in gcs_data.keys()]
    prm = [prm_data[k]['Time (ms)'] for k in prm_data.keys()]
    sprm = [sprm_data[k]['Time (ms)'] for k in sprm_data.keys()]

    plt.subplot(2, 1, 2)
    _make_bars(gcs, prm, sprm, 0)
    plt.xlabel('Task')
    plt.ylabel('Runtime (ms)')
    plt.ylim([
        0,
        max([
            np.mean(gcs) + np.std(gcs),
            np.mean(prm) + np.std(prm),
            np.mean(sprm) + np.std(sprm)
        ]) * 1.9
    ])

    plt.subplots_adjust(hspace=0.1)


def _make_bars(gcs_data, prm_data, sprm_data, round_to=2):
    c_list_rgb = [[i / 255 for i in (0, 0, 255, 255)],
                  [i / 255 for i in (255, 191, 0, 255)],
                  [i / 255 for i in (255, 64, 0, 255)]]
    c_gcs = c_list_rgb[0]
    c_prm = c_list_rgb[1]
    c_prm_shortcut = c_list_rgb[2]

    width = 0.2
    ticks = np.arange(1, 6)

    bar_options = {
        'zorder': 3,
        'edgecolor': 'k',
    }

    text_options = {
        'va': 'bottom',
        'ha': 'center',
        'fontsize': 8,
        'rotation': 90,
    }

    gcs_mean = np.mean(gcs_data, axis=1)
    gcs_std = np.std(gcs_data, axis=1)

    prm_mean = np.mean(prm_data, axis=1)
    prm_std = np.std(prm_data, axis=1)

    sprm_mean = np.mean(sprm_data, axis=1)
    sprm_std = np.std(sprm_data, axis=1)

    plt.bar(ticks - width,
            gcs_mean,
            width,
            yerr=gcs_std,
            label='GCS',
            color=c_gcs,
            **bar_options)
    plt.bar(ticks,
            prm_mean,
            width,
            yerr=prm_std,
            label='PRM',
            color=c_prm,
            **bar_options)
    plt.bar(ticks + width,
            sprm_mean,
            width,
            yerr=sprm_std,
            label='Shortcut PRM',
            color=c_prm_shortcut,
            **bar_options)

    offset = max([max(gcs_mean), max(prm_mean), max(sprm_mean)]) / 50

    gcs_mean = np.round(gcs_mean,
                        round_to) if round_to else gcs_mean.astype(int)
    prm_mean = np.round(prm_mean,
                        round_to) if round_to else prm_mean.astype(int)
    sprm_mean = np.round(sprm_mean,
                         round_to) if round_to else sprm_mean.astype(int)
    for i, x in enumerate(ticks):
        plt.text(x - width, gcs_mean[i] + gcs_std[i] + offset, gcs_mean[i],
                 **text_options)
        plt.text(x, prm_mean[i] + prm_std[i] + offset, prm_mean[i],
                 **text_options)
        plt.text(x + width, sprm_mean[i] + sprm_std[i] + offset, sprm_mean[i],
                 **text_options)

    plt.xticks(ticks)
    plt.grid()

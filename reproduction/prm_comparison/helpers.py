import matplotlib.pyplot as plt
import os
import numpy as np

from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import (
    IllustrationProperties,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Rgba,
    RoleAssign,
    Role,
    SceneGraph
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.perception import PointCloud
from pydrake.all import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import TrajectorySource, Multiplexer, ConstantVectorSource
from pydrake.multibody.tree import RevoluteJoint
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.analysis import Simulator
from pydrake.multibody import inverse_kinematics
from pydrake.solvers import Solve

from reproduction.util import *

def ForwardKinematics(q_list):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().Add("gcs", GcsDir())

    directives_file = FindModelFile("models/iiwa14_spheres_collision_welded_gripper.yaml")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)
    [iiwa, wsg, shelf, binR, binL, table] =  models

    plant.Finalize()

    diagram = builder.Build()
    
    FKcontext = diagram.CreateDefaultContext()
    FKplant_context = plant.GetMyMutableContextFromRoot(FKcontext)
    
    X_list = []
    for q in q_list:
        plant.SetPositions(FKplant_context, q)
        X_list.append(plant.EvalBodyPoseInWorld(FKplant_context, plant.GetBodyByName("body")))

    return X_list

def InverseKinematics(q0, translation, rpy):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().Add("gcs", GcsDir())

    directives_file = FindModelFile("models/iiwa14_spheres_collision_welded_gripper.yaml")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)
    [iiwa, wsg, shelf, binR, binL, table] =  models

    plant.Finalize()

    diagram = builder.Build()
    
    IKcontext = diagram.CreateDefaultContext()
    IKplant_context = plant.GetMyMutableContextFromRoot(IKcontext)
    

    gripper_frame = plant.GetBodyByName("body").body_frame()
    ik = inverse_kinematics.InverseKinematics(plant, IKplant_context)
    ik.AddPositionConstraint(
        gripper_frame, [0, 0, 0], plant.world_frame(), 
        translation, translation)
    
    ik.AddOrientationConstraint(gripper_frame, RotationMatrix(), plant.world_frame(),
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

def lower_alpha(plant, inspector, model_instances, alpha, scene_graph):
    for model in model_instances:
        for body_id in plant.GetBodyIndices(model):
            frame_id = plant.GetBodyFrameIdOrThrow(body_id)
            geometry_ids = inspector.GetGeometries(frame_id, Role.kIllustration)
            for g_id in geometry_ids:
                prop = inspector.GetIllustrationProperties(g_id)
                new_props = IllustrationProperties(prop)
                phong = prop.GetProperty("phong", "diffuse")
                phong.set(phong.r(), phong.g(), phong.b(), alpha)
                new_props.UpdateProperty("phong", "diffuse", phong)
                scene_graph.AssignRole(plant.get_source_id(), g_id, new_props, RoleAssign.kReplace)

def combine_trajectory(traj_list, wait = 2):
    knotList = []
    time_delta = 0
    time_list = []
    for traj in traj_list:
        knots = traj.vector_values(traj.get_segment_times()).T
        knotList.append(knots)
        
        duration = traj.end_time() - traj.start_time()
        offset = 0 
        try:
            offset = time_list[-1][-1] + 0.1
        except:
            pass
        time_list.append(np.linspace(offset, duration + offset,  knots.shape[0]))
        
        #add wait time
        if wait > 0.0:
            knotList.append(knotList[-1][-1,:])
            time_list.append(np.array([time_list[-1][-1] + wait]))
            
        
    path = np.vstack(knotList).T
    time_break = np.hstack(time_list)

    return PiecewisePolynomial.FirstOrderHold(time_break, path)

def make_traj(path, speed = 2):
    t_breaks = [0]
    movement_between_segment = np.sqrt(np.sum(np.square(path.T[1:,:] - path.T[:-1,:]), axis = 1))
    for s in movement_between_segment/speed:
        t_breaks += [s + t_breaks[-1]]
    return PiecewisePolynomial.FirstOrderHold(t_breaks, path)


def get_traj_length(trajectory, bspline = False, weights = None):
    path_length = 0
    if bspline:
        knots = trajectory.vector_values(np.linspace(trajectory.start_time(), trajectory.end_time(), 1000))
    else:
        knots = trajectory.vector_values(trajectory.get_segment_times())
    
    individual_mov = []
    if weights is None:
        weights = np.ones(knots.shape[0])
    for ii in range(knots.shape[1] - 1):
        path_length += np.sqrt(np.square(knots[:, ii+1] - knots[:, ii]).dot(weights))
        individual_mov.append([np.linalg.norm(knots[j, ii+1] - knots[j, ii]) for j in range(7)])
    
    return path_length

def is_traj_confined(trajectory, regions):
    knots = trajectory.vector_values(np.linspace(trajectory.start_time(), trajectory.end_time(), 1000)).T
    not_contained_knots = list(filter(lambda knot: any([r.PointInSet(knot) for r in regions.values()]), knots))

    return len(not_contained_knots)/len(knots)

def visualize_trajectory(meshcat, traj_list, show_line = False, iiwa_ghosts = [], alpha = 0.5, regions = []):
    if not isinstance(traj_list, list):
        traj_list = [traj_list]
    
    combined_traj = combine_trajectory(traj_list)

    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    plant = MultibodyPlant(time_step=0.0)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    inspector = scene_graph.model_inspector()
    
    
    parser = Parser(plant, scene_graph)
    parser.package_map().Add("gcs", GcsDir())

    directives_file = FindModelFile("models/iiwa14_spheres_collision_welded_gripper.yaml")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)
    [iiwa, wsg, shelf, binR, binL, table] =  models
    
    #add clones versions of the iiwa
    if len(iiwa_ghosts):
        lower_alpha(plant, inspector, [iiwa.model_instance, wsg.model_instance], alpha,scene_graph)
    visual_iiwas = []
    visual_wsgs = []
    iiwa_file = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_collision.urdf")
    wsg_file = FindModelFile("models/schunk_wsg_50_welded_fingers.sdf")
    
    for i, q in enumerate(iiwa_ghosts):
        new_iiwa = parser.AddModelFromFile(iiwa_file, "vis_iiwa_"+str(i))
        new_wsg = parser.AddModelFromFile(wsg_file, "vis_wsg_"+str(i))
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base", new_iiwa), RigidTransform())
        plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", new_iiwa), plant.GetFrameByName("body", new_wsg),
                         RigidTransform(rpy=RollPitchYaw([np.pi/2., 0, 0]), p=[0, 0, 0.114]))
        visual_iiwas.append(new_iiwa)
        visual_wsgs.append(new_wsg)
        lower_alpha(plant, inspector, [new_iiwa, new_wsg], alpha, scene_graph)
        index = 0
        for joint_index in plant.GetJointIndices(visual_iiwas[i]):
            joint = plant.get_mutable_joint(joint_index)
            if isinstance(joint, RevoluteJoint):
                joint.set_default_angle(q[index])
                index += 1
    
    plant.Finalize()

    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.get_source_id()))

    traj_system = builder.AddSystem(TrajectorySource(combined_traj))

    mux = builder.AddSystem(Multiplexer([7 for _ in range(1 + len(iiwa_ghosts))]))
    builder.Connect(traj_system.get_output_port(), mux.get_input_port(0))
    
    for i, q in enumerate(iiwa_ghosts):
        ghost_pos = builder.AddSystem(ConstantVectorSource(q))
        builder.Connect(ghost_pos.get_output_port(), mux.get_input_port(1+i) )
    
    
    builder.Connect(mux.get_output_port(), to_pose.get_input_port())


    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.delete_on_initialization_event = False
    meshcat_params.role = Role.kIllustration
    # meshcat_params.role = Role.kProximity
    meshcat_cpp = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, meshcat_params)
    meshcat.Delete()

    diagram = builder.Build()
    
    if show_line:
        X_lists = []
        for traj in traj_list:
            #X_list =  [ForwardKinematics(k) for k in traj.vector_values(traj.get_segment_times()).T.tolist()]
            X_list = ForwardKinematics(traj.vector_values(np.linspace(traj.start_time(), traj.end_time(), 15000)).T.tolist())
            X_lists.append(X_list)
            
        c_list_rgb = [[i/255 for i in (0, 0, 255, 255)],[i/255 for i in (255, 191, 0, 255)],[i/255 for i in (255, 64, 0, 255)]]

        for i, X_list in enumerate(X_lists):
            pointcloud = PointCloud(len(X_list))
            pointcloud.mutable_xyzs()[:] = np.array(list(map(lambda X: X.translation(), X_list))).T[:]
            meshcat.SetObject("paths/" + str(i), pointcloud, 0.015,
                              rgba=Rgba(*c_list_rgb[i]))
        
        
        reg_colors = plt.cm.viridis(np.linspace(0, 1,len(regions)))
        reg_colors[:,3] = 1.0
        
        for i, reg in enumerate(regions):
            X_reg = ForwardKinematics(spider_web(reg))
            pointcloud = PointCloud(len(reg))
            pointcloud.mutable_xyzs()[:] = np.array(list(map(lambda X: X.translation(), X_reg))).T[:]
            meshcat.SetObject("regions/" + str(i), pointcloud, 0.015,
                              rgba=Rgba(*reg_colors[i]))
        
    simulator = Simulator(diagram)
    meshcat_cpp.StartRecording()
    simulator.AdvanceTo(combined_traj.end_time())
    meshcat_cpp.PublishRecording()

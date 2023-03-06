import numpy as np
import os
import time
from copy import copy

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import (
    CollisionFilterDeclaration,
    GeometrySet,
    MeshcatVisualizer,
    Rgba,
    Role,
    SceneGraph
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.perception import PointCloud
from pydrake.solvers import MosekSolver, Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import TrajectorySource
from pydrake.systems.rendering import MultibodyPositionToGeometryPose

from gcs.bezier import BezierGCS
from gcs.linear import LinearGCS
from gcs.rounding import *
from reproduction.prm_comparison.helpers import lower_alpha
from reproduction.util import *

def getIkSeeds():
    return {
    "top_shelf/top_shelf": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.9]),
                            RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.9])),
    "top_shelf/shelf_1": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.9]),
                          RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.65])),
    "top_shelf/shelf_2": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.9]),
                        RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.4])),
    "top_shelf/bin_L": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.9]),
                        RigidTransform(RollPitchYaw(-np.pi/2+0.1, 0, np.pi), [0., 1.1, 0.3])),
    
    "shelf_1/top_shelf": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.65]),
                          RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.9])),
    "shelf_1/shelf_1": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.65]),
                        RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.65])),
    "shelf_1/shelf_2": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.65]),
                        RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.4])),
    "shelf_1/bin_L": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.65]),
                      RigidTransform(RollPitchYaw(-np.pi/2+0.1, 0, np.pi), [0., 1.1, 0.3])),
    
    "shelf_2/top_shelf": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.4]),
                          RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.9])),
    "shelf_2/shelf_1": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.4]),
                        RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.65])),
    "shelf_2/shelf_2": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.4]),
                        RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.4])),
    "shelf_2/bin_L": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.4]),
                      RigidTransform(RollPitchYaw(-np.pi/2+0.1, 0, np.pi), [0., 1.1, 0.3])),
    
    "bin_R/top_shelf": (RigidTransform(RollPitchYaw(-np.pi/2+0.1, 0, -np.pi), [0.0, -0.6, 0.3]),
                        RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.9])),
    "bin_R/shelf_1": (RigidTransform(RollPitchYaw(-np.pi/2+0.1, 0, -np.pi), [0.0, -0.6, 0.3]),
                      RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.65])),
    "bin_R/shelf_2": (RigidTransform(RollPitchYaw(-np.pi/2+0.1, 0, -np.pi), [0.0, -0.6, 0.3]),
                      RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.4])),
    "bin_R/bin_L": (RigidTransform(RollPitchYaw(-np.pi/2+0.1, 0, -np.pi), [0.0, -0.6, 0.3]),
                    RigidTransform(RollPitchYaw(-np.pi/2+0.1, 0, np.pi), [0., 1.1, 0.3])),

    "top_shelf/shelf_1_extract": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.9]),
                                  RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.5, 0.35, 0.65])),
    "top_shelf/shelf_2_extract": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.9]),
                                  RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.5, 0.35, 0.4])),
    "shelf_2_extract/top_shelf": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.5, 0.15, 0.4]),
                                  RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.9])),
    "shelf_1_extract/top_shelf": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.5, 0.15, 0.65]),
                                  RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.35, 0.9])),
    
    "top_shelf/shelf_1_cross": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.9]),
                                RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2-0.3), [0.7, 0.15, 0.65])),
    "cross_table/top_shelf_cross": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi), [0.4, 0.4, 0.2]),
                                    RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.15, 0.9])),
    "shelf_2_cross/top_shelf_cross": (RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2+0.4), [0.7, 0.35, 0.4]),
                                      RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2-0.4), [0.7, 0.15, 0.9])),
}

def getConfigurationSeeds():
    return {
        "top_shelf/top_shelf": [0.37080011,  0.41394084, -0.16861973, -0.70789778, -0.37031516, 0.60412162,  0.39982981,
                                -0.37080019,  0.41394089,  0.16861988, -0.70789766,  0.37031506,  0.60412179, -0.39982996],
        "top_shelf/shelf_1": [0.37080079,  0.41394132, -0.16862043, -0.70789679, -0.37031656, 0.60412327,  0.39982969,
                            -0.93496924,  0.46342534,  0.92801666, -1.45777635, -0.31061724, -0.0657716, -0.06019899],
        "top_shelf/shelf_2": [0.37086448,  0.41394538, -0.16875166, -0.70789745, -0.37020563, 0.60411217,  0.399785,
                            -0.4416204 ,  0.62965228,  0.20598405, -1.73324339, -0.41354372, -0.68738414,  0.17443976],
        "top_shelf/bin_L": [0.37081989,  0.41394235, -0.16866012, -0.70789737, -0.37028201, 0.60411923,  0.39981634,
                            -0.89837331, -1.1576151 ,  1.75505216, -1.37515153,  1.0676443 ,  1.56371166, -0.64126346],
        
        "shelf_1/top_shelf": [0.93496924,  0.46342534, -0.92801666, -1.45777635,  0.31061724, -0.0657716 ,  0.06019899,
                            -0.37080079,  0.41394132,  0.16862043, -0.70789679,  0.37031656,  0.60412327, -0.39982969],
        "shelf_1/shelf_1": [0.87224109,  0.43096634, -0.82223436, -1.45840049,  0.73813452, -0.08999384, -0.41624203,
                            -0.87556489,  0.43246906,  0.82766047, -1.45838515, -0.72259842, -0.0884963,  0.39840129],
        "shelf_1/shelf_2": [0.93496866,  0.463425  , -0.92801564, -1.45777634,  0.3106235, -0.06577172,  0.06019173,
                            -0.44158858,  0.62964838,  0.20594112, -1.73324341, -0.41354987, -0.6873923 ,  0.17446778],
        "shelf_1/bin_L": [0.93496918,  0.46342531, -0.92801656, -1.45777637,  0.31061728, -0.06577167,  0.06019927,
                        -0.89837321, -1.15761746,  1.75504915, -1.37515113,  1.06764716,  1.56371454, -0.64126383],
        
        "shelf_2/top_shelf": [0.4416204,  0.62965228, -0.20598405, -1.73324339,  0.41354372, -0.68738414, -0.17443976,
                            -0.37086448,  0.41394538,  0.16875166, -0.70789745,  0.37020563,  0.60411217, -0.399785],
        "shelf_2/shelf_1": [0.44158858,  0.62964838, -0.20594112, -1.73324341,  0.41354987, -0.6873923, -0.17446778,
                            -0.93496866,  0.463425  ,  0.92801564, -1.45777634, -0.3106235 , -0.06577172, -0.06019173],
        "shelf_2/shelf_2": [0.44161313,  0.62965141, -0.20597435, -1.73324346,  0.41354447, -0.68738613, -0.17444557,
                            -0.4416132 ,  0.62965142,  0.20597452, -1.73324348, -0.41354416, -0.68738609,  0.17444625],
        "shelf_2/bin_L": [0.44161528,  0.62965169, -0.20597726, -1.73324347,  0.41354399, -0.68738565, -0.17444283,
                        -1.37292761, -0.68372976,  2.96705973, -1.41521783,  2.96705973, -1.11343251, -3.0140737 ],
        
        "bin_R/top_shelf": [0.81207926, -1.25359738, -1.58098625, -1.5155474 , -1.32223687, 1.50549708, -2.38221725,
                            -0.37085114,  0.4139444 ,  0.16872443, -0.70789757,  0.37022786,  0.60411401, -0.39979449],
        "bin_R/shelf_1": [0.81207923, -1.25358454, -1.58100042, -1.51554769, -1.32222337, 1.50548369, -2.3822204 ,
                        -0.9349716 ,  0.46342674,  0.92802082, -1.45777624, -0.31059455, -0.0657707 , -0.06022391],
        "bin_R/shelf_2": [0.81207937, -1.25360462, -1.58097816, -1.51554761, -1.32224557, 1.50550485, -2.38221483,
                        -0.44166552,  0.62965782,  0.20604497, -1.7332434 , -0.41353464, -0.6873727 ,  0.17439863],
        "bin_R/bin_L": [-1.73637519,  0.6209681 ,  0.24232887, -1.51538355, -0.17977474, 0.92618894, -3.01360257,
                        1.31861497,  0.72394333,  0.4044295 , -1.37509496, -0.27461997,  1.20038493,  0.18611701],
        
        "neutral/neutral": [0.0, -0.2, 0, -1.2, 0, 1.6, 0.0, 0.0, -0.2, 0, -1.2, 0, 1.6, 0.0],

        "neutral/shelf_1": [0.0, -0.2, 0, -1.2, 0, 1.6, 0.0,
                            -0.93496866,  0.463425  ,  0.92801564, -1.45777634, -0.3106235 , -0.06577172, -0.06019173],
        "neutral/shelf_2": [0.0, -0.2, 0, -1.2, 0, 1.6, 0.0,
                            -0.44166552,  0.62965782,  0.20604497, -1.7332434 , -0.41353464, -0.6873727 ,  0.17439863],
        "shelf_1/neutral": [0.93496924,  0.46342534, -0.92801666, -1.45777635,  0.31061724, -0.0657716 ,  0.06019899,
                            0.0, -0.2, 0, -1.2, 0, 1.6, 0.0],
        "shelf_2/neutral": [0.44161528,  0.62965169, -0.20597726, -1.73324347,  0.41354399, -0.68738565, -0.17444283,
                            0.0, -0.2, 0, -1.2, 0, 1.6, 0.0],
        
        "shelf_2_cross/top_shelf_cross": [0.47500706,  0.72909874,  0.01397772, -1.52841372,  0.15392366, -0.591641, -0.12870521,
                                        -0.48821156,  0.67762534,  0.02049926, -0.27420758,  0.10620709,  0.72215209, -0.09973172],
    }

#  Additional seed points not needed to connect the graph
#     "neutral/shelf_1_extract": [ 0.0, -0.2, 0, -1.2, 0, 1.6, 0.0, -0.35486829, -0.10621117, -0.09276445, -1.94995786,  1.88826556,  0.46922151, -1.98267349],
#     "neutral/shelf_2_extract": [ 0.0, -0.2, 0, -1.2, 0, 1.6, 0.0, 0.3078069 ,  0.56765359, -0.86829439, -2.0943951 ,  2.53950045,  1.09607546, -2.4169564],
#     "shelf_1_extract/neutral": [-1.05527083, -0.43710629,  1.15648812, -1.95011062,  0.24422131, -0.07820216,  0.15872416, 0.0, -0.2, 0, -1.2, 0, 1.6, 0.0],
#     "shelf_2_extract/neutral": [-0.30739053,  0.5673891 ,  0.86772198, -2.0943951 , -2.53946773, 1.09586777,  2.41729532, 0.0, -0.2, 0, -1.2, 0, 1.6, 0.0],
#     "cross_table/top_shelf_cross": [ 0.04655887,  0.97997658,  0.52004246, -1.91926412, -1.37518707, -0.88823968,  0.07674699, -0.5921624 ,  0.83651867,  0.20513136, -0.00257881,  0.51748756,  0.92012332, -0.51686487],

def getDemoConfigurations():
    return [
        [0.0, -0.2, 0, -1.2, 0, 1.6, 0.0, 0.0, -0.2, 0, -1.2, 0, 1.6, 0.0],
        [0.69312848,  0.36303784, -0.66625368, -1.49515991,  0.3230085, -0.10942887, -0.09496304,
         -0.69312891,  0.36303794,  0.66625426, -1.49515975, -0.32300928, -0.10942832,  0.0949629],
        [0.2014604,  0.66463495,  0.16799372, -1.66212763, -0.09131682, -0.64368844, -0.03645568,
         -0.38777291,  0.56141139, -0.05760515, -0.47447495,  0.06515541,  0.63627899, -0.02552148],
        [-1.8487163 ,  0.71749397,  0.66464618, -1.4912954 , -0.52882233, 1.0096015 , -2.62844995,
         1.43620829,  0.70451542, -0.01532988, -1.34999693, -0.00550105,  1.18684923, -0.14400234],
    ]

def generateDemoConfigurations(plant, context, wsg1_id, wsg2_id):
    demo_q = [[0.0, -0.2, 0, -1.2, 0, 1.6, 0.0, 0.0, -0.2, 0, -1.2, 0, 1.6, 0.0]]

    initial_guess = copy(demo_q[0])
    demo_q.append(runBimanualIK(
            plant, context, wsg1_id, wsg2_id,
            RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.10, 0.65]),
            RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2), [0.7, 0.40, 0.65]),
            initial_guess, (0.01, 0.01)))
    
    demo_q.append(runBimanualIK(
            plant, context, wsg1_id, wsg2_id,
            RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2+0.4), [0.7, 0.25, 0.4]),
            RigidTransform(RollPitchYaw(-np.pi+0.1, 0, np.pi/2-0.4), [0.7, 0.20, 0.9]),
            initial_guess, None))
    initial_guess[0] = -np.pi/2
    initial_guess[7] = np.pi/2
    demo_q.append(runBimanualIK(
            plant, context, wsg1_id, wsg2_id,
            RigidTransform(RollPitchYaw(-np.pi/2+0.1, 0, -np.pi), [0.09, -0.6, 0.3]),
            RigidTransform(RollPitchYaw(-np.pi/2+0.1, 0, np.pi), [0.09, 1.1, 0.3]),
            initial_guess, None))
    return demo_q

def filterCollsionGeometry(scene_graph, context):
    filter_manager = scene_graph.collision_filter_manager(context)
    inspector = scene_graph.model_inspector()

    iiwa1 = [[], [], [], [], [], [], [], []]
    iiwa2 = [[], [], [], [], [], [], [], []]
    wsg1 = []
    wsg2 = []
    shelf = []
    bins = [[], []]
    table = []
    for gid in inspector.GetGeometryIds(
            GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity):
        gid_name = inspector.GetName(inspector.GetFrameId(gid))
        if "iiwa_1::iiwa_link_" in gid_name:
            link_num = gid_name[18]
            iiwa1[int(link_num)].append(gid)
        elif "iiwa_2::iiwa_link_" in gid_name:
            link_num = gid_name[18]
            iiwa2[int(link_num)].append(gid)
        elif "wsg_1" in gid_name:
            wsg1.append(gid)
        elif "wsg_2" in gid_name:
            wsg2.append(gid)
        elif "shelves::" in gid_name:
            shelf.append(gid)
        elif "binR" in gid_name:
            bins[0].append(gid)
        elif "binL" in gid_name:
            bins[1].append(gid)
        elif "table" in gid_name:
            table.append(gid)
        else:
            print("Geometry", gid_name, "not assigned to an object.")


    filter_manager.Apply(CollisionFilterDeclaration().ExcludeWithin(
            GeometrySet(iiwa1[0] + iiwa1[1] + iiwa1[2] + iiwa1[3] + shelf)))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa1[1] + iiwa1[2]+ iiwa1[3]),
            GeometrySet(iiwa1[4] + iiwa1[5])))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa1[3] + iiwa1[4]), GeometrySet(iiwa1[6])))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa1[2] + iiwa1[3] + iiwa1[4] + iiwa1[5] + iiwa1[6]),
            GeometrySet(iiwa1[7] + wsg1)))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa1[0] + iiwa1[0] + iiwa1[2]), GeometrySet(bins[0])))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa1[0] + iiwa1[1] + iiwa1[2] + iiwa1[3] + iiwa1[4]),
            GeometrySet(bins[1])))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa1[0] + iiwa1[0] + iiwa1[2]), GeometrySet(table)))

    filter_manager.Apply(CollisionFilterDeclaration().ExcludeWithin(
            GeometrySet(iiwa2[0] + iiwa2[1] + iiwa2[2] + iiwa2[3] + shelf)))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa2[1] + iiwa2[2]+ iiwa2[3]),
            GeometrySet(iiwa2[4] + iiwa2[5])))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa2[3] + iiwa2[4]), GeometrySet(iiwa2[6])))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa2[2] + iiwa2[3] + iiwa2[4] + iiwa2[5] + iiwa2[6]),
            GeometrySet(iiwa2[7] + wsg2)))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa2[0] + iiwa2[0] + iiwa2[2]), GeometrySet(bins[1])))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa2[0] + iiwa2[1] + iiwa2[2] + iiwa2[3] + iiwa2[4]),
            GeometrySet(bins[0])))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa2[0] + iiwa2[0] + iiwa2[2]), GeometrySet(table)))

    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa1[0] + iiwa1[1]), GeometrySet(iiwa2[0] + iiwa2[1])))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa1[2]), GeometrySet(iiwa2[0] + iiwa2[1])))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(iiwa1[0] + iiwa1[1]), GeometrySet(iiwa2[2])))

    pairs = scene_graph.get_query_output_port().Eval(context).inspector().GetCollisionCandidates()
    print("Filtered collision pairs from",
          len(inspector.GetCollisionCandidates()), "to", len(pairs))

# initial_guess = np.concatenate((q0, q0))
# min_dist = (0.01, 0.01)???
def runBimanualIK(plant, context, wsg1_id, wsg2_id, wsg1_pose, wsg2_pose,
                  initial_guess, min_dist=None):
    hand_frame1 = plant.GetBodyByName("body", wsg1_id).body_frame()
    hand_frame2 = plant.GetBodyByName("body", wsg2_id).body_frame()

    ik = InverseKinematics(plant, context)
    if min_dist is not None:
        ik.AddMinimumDistanceConstraint(*min_dist)
    ik.prog().AddBoundingBoxConstraint(plant.GetPositionLowerLimits(),
                                       plant.GetPositionUpperLimits(), ik.q())
    ik.prog().SetInitialGuess(ik.q(), initial_guess)
    ik.prog().AddQuadraticCost((ik.q() - initial_guess).dot(ik.q() - initial_guess))

    ik.AddPositionConstraint(hand_frame1, [0, 0, 0], plant.world_frame(),
                             wsg1_pose.translation(), wsg1_pose.translation())
    ik.AddOrientationConstraint(hand_frame1, RotationMatrix(), plant.world_frame(),
                                wsg1_pose.rotation(), 0.001)

    ik.AddPositionConstraint(hand_frame2, [0, 0, 0], plant.world_frame(),
                             wsg2_pose.translation(), wsg2_pose.translation())
    ik.AddOrientationConstraint(hand_frame2, RotationMatrix(), plant.world_frame(),
                                wsg2_pose.rotation(), 0.001)

    result = Solve(ik.prog())
    return result.GetSolution(ik.q())

def visualizeConfig(diagram, plant, context, q):
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, q)
    diagram.Publish(context)

def getLinearGcsPath(regions, sequence):
    path = [sequence[0]]
    run_time = 0.0
    gcs = LinearGCS(regions)
    gcs.setPaperSolverOptions()
    gcs.setSolver(MosekSolver())
    for start_pt, goal_pt in zip(sequence[:-1], sequence[1:]):
        gcs.addSourceTarget(start_pt, goal_pt)
        
        start_time = time.time()
        waypoints, results_dict = gcs.SolvePath(True, False, preprocessing=True)
        if waypoints is None:
            print(f"Failed between {start_pt} and {goal_pt}")
            return None
        print(f"Planned segment in {np.round(time.time() - start_time, 4)}", flush=True)
        # run_time += results_dict["preprocessing_stats"]['linear_programs']
        run_time += results_dict["relaxation_solver_time"]
        run_time += results_dict["total_rounded_solver_time"]
        
        path += waypoints.T[1:].tolist()

        gcs.ResetGraph()
    
    return np.stack(path).T, run_time

def getBezierGcsPath(plant, regions, sequence, order, continuity, hdot_min = 1e-3):
    run_time = []
    trajectories = []
    gcs = BezierGCS(regions, order, continuity)
    gcs.addTimeCost(1)
    gcs.addPathLengthCost(1)
    gcs.addDerivativeRegularization(1e-3, 1e-3, 2)
    gcs.addVelocityLimits(0.6*plant.GetVelocityLowerLimits(), 0.6*plant.GetVelocityUpperLimits())
    gcs.setPaperSolverOptions()
    gcs.setSolver(MosekSolver())
    gcs.setRoundingStrategy(randomForwardPathSearch, max_paths = 10, max_trials = 100, seed = 0)
    for start_pt, goal_pt in zip(sequence[:-1], sequence[1:]):
        segment_run_time=0.0
        gcs.addSourceTarget(start_pt, goal_pt)
        
        start_time = time.time()
        segment_traj, results_dict = gcs.SolvePath(True, False, preprocessing=True)
        if segment_traj is None:
            print(f"Failed between {start_pt} and {goal_pt}")
            return None
        print(f"Planned segment in {np.round(time.time() - start_time, 4)}", flush=True)
        # segment_run_time += results_dict["preprocessing_stats"]['linear_programs']
        segment_run_time += results_dict["relaxation_solver_time"]
        segment_run_time += results_dict["total_rounded_solver_time"]
        trajectories.append(segment_traj)
        run_time.append(segment_run_time)
        print("\tRounded cost:", np.round(results_dict["rounded_cost"], 4),
              "\tRelaxed cost:", np.round(results_dict["relaxation_cost"], 4))
        print("\tCertified Optimality Gap:",
             (results_dict["rounded_cost"]-results_dict["relaxation_cost"])
                /results_dict["relaxation_cost"])

        gcs.ResetGraph()
        
    return trajectories, run_time

class VectorTrajectorySource(LeafSystem):
    def __init__(self, trajectories):
        LeafSystem.__init__(self)
        self.trajectories = trajectories
        self.start_time = [0]
        for traj in trajectories:
            self.start_time.append(self.start_time[-1] + traj.end_time())
        self.start_time = np.array(self.start_time)
        self.port = self.DeclareVectorOutputPort("traj_eval", 14, self.DoVecTrajEval, {self.time_ticket()})
    
    def DoVecTrajEval(self, context, output):
        t = context.get_time()
        traj_index = np.argmax(self.start_time > t) - 1
        
        q = self.trajectories[traj_index].value(t - self.start_time[traj_index])
        output.set_value(q)

def visualize_trajectory(traj, meshcat):
    builder = DiagramBuilder()

    scene_graph = builder.AddSystem(SceneGraph())
    plant = MultibodyPlant(time_step=0.0)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    parser = Parser(plant)
    parser.package_map().Add("gcs", GcsDir())

    directives_file = FindModelFile("models/bimanual_iiwa.yaml")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)
    [iiwa_1, wsg_1, iiwa_2, wsg_2, shelf, binR, binL, table] =  models

    plant.Finalize()

    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.get_source_id()))

    if type(traj) is list:
        traj_system = builder.AddSystem(VectorTrajectorySource(traj))
        end_time = np.sum([t.end_time() for t in traj])
    else:
        traj_system = builder.AddSystem(TrajectorySource(traj))
        end_time = traj.end_time()
    builder.Connect(traj_system.get_output_port(), to_pose.get_input_port())

    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.Delete()

    vis_diagram = builder.Build()
    simulator = Simulator(vis_diagram)

    plant_context = plant.CreateDefaultContext()
    rgb_color = [i/255 for i in (0, 0, 255, 255)]
    iiwa1_X = []
    iiwa2_X = []
    if type(traj) is list:
        for t in traj:
            q_waypoints = t.vector_values(np.linspace(t.start_time(), t.end_time(), 1000))
            for ii in range(q_waypoints.shape[1]):
                plant.SetPositions(plant_context, q_waypoints[:, ii])
                iiwa1_X.append(plant.EvalBodyPoseInWorld(
                    plant_context, plant.GetBodyByName("body", wsg_1.model_instance)))
                iiwa2_X.append(plant.EvalBodyPoseInWorld(
                    plant_context, plant.GetBodyByName("body", wsg_2.model_instance)))

        iiwa1_pointcloud = PointCloud(len(iiwa1_X))
        iiwa1_pointcloud.mutable_xyzs()[:] = np.array(
                list(map(lambda X: X.translation(), iiwa1_X))).T[:]
        meshcat.SetObject("paths/iiwa_1", iiwa1_pointcloud, 0.015,
                            rgba=Rgba(*rgb_color))
        iiwa2_pointcloud = PointCloud(len(iiwa2_X))
        iiwa2_pointcloud.mutable_xyzs()[:] = np.array(
                list(map(lambda X: X.translation(), iiwa2_X))).T[:]
        meshcat.SetObject("paths/iiwa_2", iiwa2_pointcloud, 0.015,
                            rgba=Rgba(*rgb_color))

    meshcat_viz.StartRecording()
    simulator.AdvanceTo(end_time)
    meshcat_viz.PublishRecording()

def generate_segment_pics(traj, segment, meshcat):
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant, scene_graph)
    parser.package_map().Add("gcs", GcsDir())
    directives_file = FindModelFile("models/bimanual_iiwa.yaml")
    iiwa_file = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_collision.urdf")
    wsg_file = FindModelFile("models/schunk_wsg_50_welded_fingers.sdf")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)
    [iiwa1_start, wsg1_start, iiwa2_start, wsg2_start, shelf, binR, binL, table] =  models

    iiwa1_goal = parser.AddModelFromFile(iiwa_file, "iiwa1_goal")
    wsg1_goal = parser.AddModelFromFile(wsg_file, "wsg1_goal")
    iiwa2_goal = parser.AddModelFromFile(iiwa_file, "iiwa2_goal")
    wsg2_goal = parser.AddModelFromFile(wsg_file, "wsg2_goal")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base", iiwa1_goal),
                     RigidTransform())
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa1_goal),
                     plant.GetFrameByName("body", wsg1_goal),
                     RigidTransform(rpy=RollPitchYaw([np.pi/2., 0, np.pi/2]), p=[0, 0, 0.114]))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base", iiwa2_goal),
                     RigidTransform([0, 0.5, 0]))
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa2_goal),
                     plant.GetFrameByName("body", wsg2_goal),
                     RigidTransform(rpy=RollPitchYaw([np.pi/2., 0, np.pi/2]), p=[0, 0, 0.114]))

    arm_models = [iiwa1_start.model_instance, wsg1_start.model_instance,
                  iiwa2_start.model_instance, wsg2_start.model_instance,
                  iiwa1_goal, wsg1_goal, iiwa2_goal, wsg2_goal]
    lower_alpha(plant, scene_graph.model_inspector(), arm_models, 0.4, scene_graph)

    plant.Finalize()

    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    meshcat.Delete()

    diagram = builder.Build()

    if type(traj) is not list:
        traj = [traj]

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    t = traj[segment]

    iiwa1_X = []
    iiwa2_X = []
    q_waypoints = t.vector_values(np.linspace(t.start_time(), t.end_time(), 1000))
    for ii in range(q_waypoints.shape[1]):
        plant.SetPositions(plant_context, np.concatenate((q_waypoints[:, ii], np.zeros(14))))
        iiwa1_X.append(plant.EvalBodyPoseInWorld(
            plant_context, plant.GetBodyByName("body", wsg1_start.model_instance)))
        iiwa2_X.append(plant.EvalBodyPoseInWorld(
            plant_context, plant.GetBodyByName("body", wsg2_start.model_instance)))

        iiwa1_pointcloud = PointCloud(len(iiwa1_X))
        iiwa1_pointcloud.mutable_xyzs()[:] = np.array(
                list(map(lambda X: X.translation(), iiwa1_X))).T[:]
        meshcat.SetObject("paths/iiwa_1", iiwa1_pointcloud, 0.015,
                            rgba=Rgba(0, 0, 1, 1))
        iiwa2_pointcloud = PointCloud(len(iiwa2_X))
        iiwa2_pointcloud.mutable_xyzs()[:] = np.array(
                list(map(lambda X: X.translation(), iiwa2_X))).T[:]
        meshcat.SetObject("paths/iiwa_2", iiwa2_pointcloud, 0.015,
                            rgba=Rgba(0, 0, 1, 1))

    plant.SetPositions(plant_context, np.concatenate((q_waypoints[:, 0], q_waypoints[:, -1])))
    diagram.Publish(context)

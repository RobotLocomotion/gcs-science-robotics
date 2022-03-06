def ForwardKinematics(q_list):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().Add( "wsg_50_description", os.path.dirname(FindResourceOrThrow(
            "drake/manipulation/models/wsg_50_description/package.xml")))


    directives_file = FindResourceOrThrow("drake/planning/models/iiwa14_spheres_collision_welded_gripper.yaml")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)
    [iiwa, wsg, shelf, binR, binL] =  models

    plant.Finalize()

    diagram = builder.Build()
    
    FKcontext = diagram.CreateDefaultContext()
    FKplant_context = plant.GetMyMutableContextFromRoot(FKcontext)
    
    X_list = []
    for q in q_list:
        plant.SetPositions(FKplant_context, q)
        X_list.append(plant.EvalBodyPoseInWorld(FKplant_context, plant.GetBodyByName("body")))

    return X_list


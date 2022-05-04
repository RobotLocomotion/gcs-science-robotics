import numpy as np
import matplotlib.pyplot as plt
import lxml.etree as ET

from pydrake.geometry.optimization import HPolyhedron
from pydrake.math import RigidTransform, RollPitchYaw

def generate_grid_world(shape, start, goal, seed=None):
    '''
        shape: 2-tuple (rows, cols)
        start: 2-element np array of integer index of start cell
        goal: 2-element np array of integer index of goal cell
        seed: rng seed for np or None
    '''

    if seed is not None:
        np.random.seed(seed)

    # Generate room layout on a grid via a simple growth algorithm.
    # States:
    #  -1: Undetermined
    #   0: Outside
    #   1: Inside
    grid = np.zeros(shape) - 1
    # Make the goal always inside, and the start always outside.
    grid[start[0], start[1]] = 0
    grid[goal[0], goal[1]] = 1

    growth_queue = [goal]
    # For each neighbor of an indoor cell, maybe grow the building
    # into it.
    while len(growth_queue) > 0:
        here = growth_queue.pop(0)
        for dx, dy in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
            target = here + np.array([dx, dy])
            # Bounds check
            if target[0] < 0 or target[1] < 0 or target[0] >= grid.shape[0] or target[1] >= grid.shape[1]:
                continue
            # Already-decided check
            if grid[target[0], target[1]] >= 0:
                continue
            grow = np.random.random() > 0.3
            if grow:
                grid[target[0], target[1]] = 1
                growth_queue.append(target)
            else:
                grid[target[0], target[1]] = 0

    # Now extract edges where we might place walls; either
    # outer wall on indoor/outdoor border, or obstacles
    # inside the building.
    indoor_edges = []
    outdoor_edges = []
    for i in range(-1, grid.shape[0]):
        for j in range(-1, grid.shape[1]):
            # Look in the +x and +y directions, when they're in bound.
            first = np.array([i, j])
            if i < 0 or j < 0:
                first_state = 0
            else:
                first_state = grid[i, j]
            for dx, dy in zip([1, 0], [0, 1]):
                second = first + np.array([dx, dy])
                if second[0] < 0 or second[1] < 0 or second[0] >= grid.shape[0] or second[1] >= grid.shape[1]:
                    second_state = 0
                else:
                    second_state = grid[second[0], second[1]]
                midpoint = (first + second)/2.
                wall_endpoints = [midpoint + np.array([-dy, dx])/2., midpoint + np.array([dy, -dx])/2.]
                if first_state > 0.5 and np.isclose(first_state, second_state):
                    # Both indoors.
                    indoor_edges.append(wall_endpoints)
                elif first_state > 0.5 and second_state < 0.5:
                    # Indoor-outdoor transition going one way.
                    outdoor_edges.append(wall_endpoints)
                elif first_state < 0.5 and second_state > 0.5:
                    # Indoor-outdoor transition going other way.
                    outdoor_edges.append(wall_endpoints[::-1])

    return grid, indoor_edges, outdoor_edges

def draw_grid_world(grid, start, goal, indoor_edges, outdoor_edges):
    plt.figure(dpi=300).set_size_inches(3, 6)
    # Plot the grid
    grid = np.clip(grid, 0, 1)
    plt.imshow(grid.T, cmap="binary", vmin=0, vmax=1)
    # Plot start and goal
    plt.scatter(np.ones(1)*start[0], np.ones(1)*start[1], s=100, marker="+")
    plt.scatter(np.ones(1)*goal[0], np.ones(1)*goal[1], s=100, marker="*")

    # Draw those walls.
    for e1, e2 in indoor_edges:
        plt.plot([e1[0], e2[0]], [e1[1], e2[1]], linestyle="--", c="red")
    for e1, e2 in outdoor_edges:
        plt.arrow(e1[0], e1[1], (e2 - e1)[0], (e2 - e1)[1], linestyle="-", color="orange", head_width=0.1)

    plt.xlim([-2, grid.shape[0]])
    plt.ylim([-2, grid.shape[1]])
    
# Compile that into a Drake scene by assembling walls, floor, and ceiling tiles together.
def compile_sdf(output_file, grid, start, goal, indoor_edges, outdoor_edges, seed=None):
    '''
        Glue together constituent SDFs into one big SDF for the whole scene.
    '''

    if seed is not None:
        np.random.seed(seed)

    # These dict list files + their relative weights of being chosen.
    indoor_options = {
        "models/room_gen/half_wall_horizontal.sdf": 0.5,
        "models/room_gen/half_wall_horizontal_mirror.sdf": 0.5,
        "models/room_gen/half_wall_vertical.sdf": 0.25,
        "models/room_gen/wall_with_center_door_internal.sdf": 0.5,
        "": 0.25,
    }
    wall_options = {
        "models/room_gen/just_wall.sdf": 1.0,
        "models/room_gen/wall_with_center_door.sdf": 0.1,
        "models/room_gen/wall_with_left_window.sdf": 0.05,
        "models/room_gen/wall_with_right_window.sdf": 0.05,
        "models/room_gen/wall_with_windows.sdf": 0.02,
    }
    tree_probability = 0.7

    root_item = ET.Element('sdf', version="1.5", nsmap={'drake': 'drake.mit.edu'})
    world_item = ET.SubElement(root_item, "world", name="building")
    model_item = ET.SubElement(root_item, "model", name="building")

    def include_static_sdf_at_pose(name, uri, tf):
        include_item = ET.SubElement(model_item, "include")
        name_item = ET.SubElement(include_item, "name")
        name_item.text = name
        uri_item = ET.SubElement(include_item, "uri")
        uri_item.text = "package://gcs/" + uri
        static_item = ET.SubElement(include_item, "static")
        static_item.text = "True"
        pose_item = ET.SubElement(include_item, "pose")
        xyz = tf.translation()
        rpy = RollPitchYaw(tf.rotation()).vector()
        pose_item.text = "%f %f %f %f %f %f" % (
            *xyz, *rpy
        )
        
    regions = []
    
    quad_radius = 0.2
    wall_offset = 0.125 + quad_radius
    z_min = quad_radius
    z_max = 3 - quad_radius
    x_cells, y_cells = grid.shape
    # Add regions for box around building
    regions = [HPolyhedron.MakeBox([-2.5, -2.5, z_min],
                                    [x_cells * 5 + 7.5, 2.5 - wall_offset, z_max]),
                HPolyhedron.MakeBox([-2.5, 2.5 - wall_offset, z_min],
                                    [2.5 - wall_offset, y_cells * 5 + 2.5 + wall_offset, z_max]),
                HPolyhedron.MakeBox([x_cells * 5 + 2.5 + wall_offset, 2.5 - wall_offset, z_min],
                                    [x_cells * 5 + 7.5, y_cells * 5 + 2.5 + wall_offset, z_max]),
                HPolyhedron.MakeBox([-2.5, y_cells * 5 + 2.5 + wall_offset, z_min],
                                    [x_cells * 5 + 7.5, y_cells * 5 + 7.5, z_max])]
        
    # Populate floor and ceilings.
    for i in range(-1, x_cells + 1):
        for j in range(-1, y_cells + 1):
            xy = (np.array([i, j]) - start)*5
            # Floor
            tf = RigidTransform(p=np.r_[xy, 0])
            # Indoors
            if i >= 0 and j >= 0 and i < x_cells and j < y_cells and grid[i, j] > 0.5:
                include_static_sdf_at_pose("floor_%05d_%05d" % (i, j), "models/room_gen/floor_indoor.sdf", tf)
                include_static_sdf_at_pose("ceiling_%05d_%05d" % (i, j), "models/room_gen/ceiling.sdf", tf)
                regions.append(HPolyhedron.MakeBox(
                    [xy[0] - (2.5 - wall_offset), xy[1] - (2.5 - wall_offset), z_min],
                    [xy[0] + (2.5 - wall_offset), xy[1] + (2.5 - wall_offset), z_max]))
            # Outdoors
            else:
                include_static_sdf_at_pose("floor_%05d_%05d" % (i, j), "models/room_gen/floor_outdoor.sdf", tf)
                if i < 0 or j < 0 or i == x_cells or j == y_cells:
                    continue
                lb = [xy[0]-2.5, xy[1]-2.5, z_min]
                ub = [xy[0]+2.5, xy[1]+2.5, z_max]
                if i == 0:
                    lb[0] -= wall_offset
                if j == 0:
                    lb[1] -= wall_offset
                if i == x_cells - 1:
                    ub[0] += wall_offset
                if j == y_cells - 1:
                    ub[1] += wall_offset
                if i > 0 and j >= 0 and j < y_cells and grid[i - 1, j] > 0.5:
                    lb[0] += wall_offset
                if j > 0 and i >= 0 and i < x_cells and grid[i, j - 1] > 0.5:
                    lb[1] += wall_offset
                if i < x_cells - 1 and j >= 0 and j < y_cells and grid[i + 1, j] > 0.5:
                    ub[0] -= wall_offset
                if j < y_cells - 1 and i >= 0 and i < x_cells and grid[i, j + 1] > 0.5:
                    ub[1] -= wall_offset

                if np.random.random() < 1 - tree_probability:
                    regions.append(HPolyhedron.MakeBox(lb, ub))
                    continue
                else:
                    tree_pose = xy + 3.0*np.random.rand(2)-1.5
                    tf = RigidTransform(p=np.r_[tree_pose, 0])
                    include_static_sdf_at_pose("tree_%05d_%05d" % (i, j), "models/room_gen/tree.sdf", tf)
                    
                    regions.append(HPolyhedron.MakeBox(lb, [ub[0], tree_pose[1] - 0.5, ub[2]]))
                    regions.append(HPolyhedron.MakeBox([lb[0], tree_pose[1] - 0.5, lb[2]],
                                                        [tree_pose[0] - 0.5, tree_pose[1] + 0.5, ub[2]]))
                    regions.append(HPolyhedron.MakeBox([tree_pose[0] + 0.5, tree_pose[1] - 0.5, lb[2]],
                                                        [ub[0], tree_pose[1] + 0.5, ub[2]]))
                    regions.append(HPolyhedron.MakeBox([lb[0], tree_pose[1] + 0.5, lb[2]], ub))

    # Wall pass through constants
    door_width = 1.25 - 2 * quad_radius
    door_height = 2 - quad_radius
    window_width = 1.5 - 2 * quad_radius
    window_offset = 1.25
    window_z_min = 0.75 + quad_radius
    window_z_max = 2.25 - quad_radius
    half_wall_offset = 1.25
    
    # Outer walls.
    key_options = list(wall_options.keys())
    probs = np.array(list(wall_options.values()))
    probs = probs / np.sum(probs)
    np.random.shuffle(outdoor_edges)
    for k, (e1, e2) in enumerate(outdoor_edges):
        # Force first wall to be a door option, so scene is traversable.
        wall_option = ""
        sdf_key = np.random.choice(key_options, p=probs)
        while (k == 0 and "door" not in sdf_key and "window" not in sdf_key):
            sdf_key = np.random.choice(key_options, p=probs)
        
        # Take their average for the centerpoint, and infer rotation from the points.
        delta = e2 - e1
        theta = np.arctan2(delta[0], delta[1])
        midpoint = (e1 + e2)/2.

        # Coordinate shift into wall blocks: "rooms" are 5m blocks, and shift
        # so start is at the origin.
        midpoint = (midpoint - start) * 5
        
        if "door" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + door_width/2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + door_width/2.0 * np.cos(theta))
            lb = np.array([midpoint[0] - dx, midpoint[1] - dy , z_min])
            ub = np.array([midpoint[0] + dx, midpoint[1] + dy , door_height])
            regions.append(HPolyhedron.MakeBox(lb, ub))
        elif "left_window" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + window_width/2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + window_width/2.0 * np.cos(theta))
            lb = np.array([midpoint[0] - dx + window_offset * np.sin(theta), midpoint[1] - dy + window_offset * np.cos(theta), window_z_min])
            ub = np.array([midpoint[0] + dx + window_offset * np.sin(theta), midpoint[1] + dy + window_offset * np.cos(theta), window_z_max])
            regions.append(HPolyhedron.MakeBox(lb, ub))
        elif "right_window" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + window_width/2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + window_width/2.0 * np.cos(theta))
            lb = np.array([midpoint[0] - dx - window_offset * np.sin(theta), midpoint[1] - dy - window_offset * np.cos(theta), window_z_min])
            ub = np.array([midpoint[0] + dx - window_offset * np.sin(theta), midpoint[1] + dy - window_offset * np.cos(theta), window_z_max])
            regions.append(HPolyhedron.MakeBox(lb, ub))
        elif "windows" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + window_width/2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + window_width/2.0 * np.cos(theta))
            lb = np.array([midpoint[0] - dx + window_offset * np.sin(theta), midpoint[1] - dy + window_offset * np.cos(theta), window_z_min])
            ub = np.array([midpoint[0] + dx + window_offset * np.sin(theta), midpoint[1] + dy + window_offset * np.cos(theta), window_z_max])
            regions.append(HPolyhedron.MakeBox(lb, ub))
            
            lb = np.array([midpoint[0] - dx - window_offset * np.sin(theta), midpoint[1] - dy - window_offset * np.cos(theta), window_z_min])
            ub = np.array([midpoint[0] + dx - window_offset * np.sin(theta), midpoint[1] + dy - window_offset * np.cos(theta), window_z_max])
            regions.append(HPolyhedron.MakeBox(lb, ub))

        tf = RigidTransform(p=np.r_[midpoint, 0], rpy=RollPitchYaw(0, 0, -theta))
        include_static_sdf_at_pose("outer_wall_%05d" % k, sdf_key, tf)

    # Inner walls.
    key_options = list(indoor_options.keys())
    probs = np.array(list(indoor_options.values()))
    probs = probs / np.sum(probs)
    np.random.shuffle(indoor_edges)
    for k, (e1, e2) in enumerate(indoor_edges):
        # Force first wall to be a door option, so scene is traversable.
        wall_option = ""
        sdf_key = np.random.choice(key_options, p=probs)
        # Take their average for the centerpoint, and infer rotation from the points.
        delta = e2 - e1
        theta = np.arctan2(*delta)
        midpoint = (e1 + e2)/2.

        # Coordinate shift into wall blocks: "rooms" are 5m blocks, and shift
        # so start is at the origin.
        midpoint = (midpoint - start) * 5
        
        if sdf_key == "":
            dx = np.abs(wall_offset * np.cos(theta) + (2.5 - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (2.5 - wall_offset) * np.cos(theta))
            lb = np.array([midpoint[0] - dx, midpoint[1] - dy , z_min])
            ub = np.array([midpoint[0] + dx, midpoint[1] + dy , z_max])
            regions.append(HPolyhedron.MakeBox(lb, ub))
            continue
        elif "door" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + door_width/2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + door_width/2.0 * np.cos(theta))
            lb = np.array([midpoint[0] - dx, midpoint[1] - dy , z_min])
            ub = np.array([midpoint[0] + dx, midpoint[1] + dy , door_height])
            regions.append(HPolyhedron.MakeBox(lb, ub))
        elif "mirror" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + (1.25 - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (1.25 - wall_offset) * np.cos(theta))
            lb = np.array([midpoint[0] - dx - half_wall_offset * np.sin(theta), midpoint[1] - dy + half_wall_offset * np.cos(theta), z_min])
            ub = np.array([midpoint[0] + dx - half_wall_offset * np.sin(theta), midpoint[1] + dy + half_wall_offset * np.cos(theta), z_max])
            regions.append(HPolyhedron.MakeBox(lb, ub))
        elif "horizontal" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + (1.25 - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (1.25 - wall_offset) * np.cos(theta))
            lb = np.array([midpoint[0] - dx + half_wall_offset * np.sin(theta), midpoint[1] - dy - half_wall_offset * np.cos(theta), z_min])
            ub = np.array([midpoint[0] + dx + half_wall_offset * np.sin(theta), midpoint[1] + dy - half_wall_offset * np.cos(theta), z_max])
            regions.append(HPolyhedron.MakeBox(lb, ub))
        elif "vertical" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + (2.5 - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (2.5 - wall_offset) * np.cos(theta))
            lb = np.array([midpoint[0] - dx, midpoint[1] - dy, 1.7])
            ub = np.array([midpoint[0] + dx, midpoint[1] + dy, z_max])
            regions.append(HPolyhedron.MakeBox(lb, ub))

        tf = RigidTransform(p=np.r_[midpoint, 0], rpy=RollPitchYaw(0, 0, theta))
        include_static_sdf_at_pose("inner_wall_%05d" % k, sdf_key, tf)

    # Start and end indicators
    tf = RigidTransform(p=np.r_[(start-start)*5, 0])
    include_static_sdf_at_pose("start_indicator", "models/room_gen/start.sdf", tf)


    tf = RigidTransform(p=np.r_[(goal-start)*5, 0])
    include_static_sdf_at_pose("goal_indicator", "models/room_gen/target.sdf", tf)

    tree = ET.ElementTree(root_item)
    tree.write(output_file, pretty_print=True)
    
    return regions

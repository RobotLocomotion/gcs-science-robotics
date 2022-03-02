import networkx as nx

### Edges from regions
def findEdgesViaOverlaps(regions):
    edges = []
    for ii in range(len(regions)):
        for jj in range(ii + 1, len(regions)):
            if regions[ii].IntersectsWith(regions[jj]):
                edges.append((ii, jj))
                edges.append((jj, ii))
    return edges

def findStartGoalEdges(regions, start, goal):
    edges = [[], []]
    for ii in range(len(regions)):
        if regions[ii].PointInSet(start):
            edges[0].append(ii)
        if regions[ii].PointInSet(goal):
            edges[1].append(ii)
    return edges

### Rounding Strategies
def greedyForwardPathSearch(spp, result, start, goal):
    # Extract path with a tree walk
    vertices = [start]
    active_edges = []
    unused_edges = spp.Edges()
    max_phi = 0
    max_edge = None
    for edge in unused_edges:
        phi = result.GetSolution(edge.phi())
        if edge.u() == start and phi > max_phi:
            max_phi = phi
            max_edge = edge
    if max_edge is None:
        return None
    active_edges.append(max_edge)
    unused_edges.remove(max_edge)
    vertices.append(max_edge.v())
    
    while active_edges[-1].v() != goal:
        max_phi = 0
        max_edge = None
        for edge in unused_edges:
            phi = result.GetSolution(edge.phi())
            if edge.u() == active_edges[-1].v() and phi > max_phi:
                max_phi = phi
                max_edge = edge
        if max_edge is None:
            return None
        active_edges.append(max_edge)
        unused_edges.remove(max_edge)
        if max_edge.v() in vertices:
            loop_index = vertices.index(max_edge.v())
            active_edges = active_edges[:loop_index]
            vertices = vertices[:loop_index+1]
        else:
            vertices.append(max_edge.v())
    return active_edges

def dijkstraRounding(gcs, result, source, target, flow_min=1e-4):
    G = nx.DiGraph()
    G.add_nodes_from(gcs.Vertices())
    G.add_edges_from([(e.u(), e.v()) for e in gcs.Edges()])

    for e in gcs.Edges():
        flow = result.GetSolution(e.phi())
        if flow > flow_min:
            G.edges[e.u(), e.v()]['l'] = e.GetSolutionCost(result) / flow

    path = nx.bidirectional_dijkstra(G, source, target, 'l')[1]

    active_edges = []
    for u, v in zip(path[:-1], path[1:]):
        for e in gcs.Edges():
            if e.u() == u and e.v() == v:
                active_edges.append(e)
                break

    return active_edges

### Solve with rounding
def solveSPP(spp, start, goal, rounding, solver=None, solver_options=None,
             rounding_fn=greedyForwardPathSearch):
    result = spp.SolveShortestPath(start, goal, rounding, solver, solver_options)
    if not result.is_success():
        print("First solve failed")
        return None, result, None

    # Extract path
    active_edges = rounding_fn(spp, result, start, goal)
    if active_edges is None:
        print("Could not find path")
        return None, result, None

    # Solve with hard edge choices
    if rounding:
        for edge in spp.Edges():
            if edge in active_edges:
                edge.AddPhiConstraint(True)
            else:
                edge.AddPhiConstraint(False)
        hard_result = spp.SolveShortestPath(start, goal, rounding, solver, solver_options)
        if not hard_result.is_success():
            print("Second solve failed")
            return None, result, hard_result
    else:
        hard_result = result
    return active_edges, result, hard_result

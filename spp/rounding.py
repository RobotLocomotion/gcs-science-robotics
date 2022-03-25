import networkx as nx

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

def greedyBackwardPathSearch(spp, result, start, goal):
    # Extract path with a tree walk
    vertices = [goal]
    active_edges = []
    unused_edges = spp.Edges()
    max_phi = 0
    max_edge = None
    for edge in unused_edges:
        phi = result.GetSolution(edge.phi())
        if edge.v() == goal and phi > max_phi:
            max_phi = phi
            max_edge = edge
    if max_edge is None:
        return None
    active_edges.insert(0, max_edge)
    unused_edges.remove(max_edge)
    vertices.insert(0, max_edge.u())

    while active_edges[0].u() != start:
        max_phi = 0
        max_edge = None
        for edge in unused_edges:
            phi = result.GetSolution(edge.phi())
            if edge.v() == active_edges[0].u() and phi > max_phi:
                max_phi = phi
                max_edge = edge
        if max_edge is None:
            return None
        active_edges.insert(0, max_edge)
        unused_edges.remove(max_edge)
        if max_edge.u() in vertices:
            loop_index = vertices.index(max_edge.u())
            active_edges = active_edges[loop_index+1:]
            vertices = vertices[loop_index:]
        else:
            vertices.insert(0, max_edge.u())
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

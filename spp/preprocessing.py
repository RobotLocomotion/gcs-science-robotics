import numpy as np
from pydrake.all import MathematicalProgram, Solve

def removeRedundancies(gcs, s, t, tol=1e-4, verbose=False):

    if verbose:
        print('Vertices before preprocessing:', len(gcs.Vertices()))
        print('Edges before preprocessing:', len(gcs.Edges()))

    # Edges incident with each vertex.
    inedges_w = lambda w: [k for k, e in enumerate(gcs.Edges()) if e.v() == w]
    outedges_w = lambda w: [k for k, e in enumerate(gcs.Edges()) if e.u() == w]

    # Ensure that s and t have no incoming and outgoing edges, respectively.
    removal_edges = []
    for k in inedges_w(s) + outedges_w(t):
        removal_edges.append(gcs.Edges()[k])
    for e in removal_edges:
        gcs.RemoveEdge(e)

    # Eliminate vertices that do not have incident edges.
    # This ensures that the conservations of flow are well defined.
    def removeIsolatedVertices():
        isolated = lambda w: len(inedges_w(w)) == 0 and len(outedges_w(w)) == 0
        isolated_vertices = [w for w in gcs.Vertices() if isolated(w)]
        if s in isolated_vertices:
            raise ValueError('Source vertex is isolated.')
        if t in isolated_vertices:
            raise ValueError('Target vertex is isolated.')
        for w in isolated_vertices:
            gcs.RemoveVertex(w)
    removeIsolatedVertices()

    # Flow from s to u.
    nE = len(gcs.Edges())
    zeroE = np.zeros(nE)
    onesE = np.ones(nE)
    prog = MathematicalProgram()
    f = prog.NewContinuousVariables(nE, 'f')
    f_limits = prog.AddBoundingBoxConstraint(zeroE, onesE, f).evaluator()

    # Flow from v to t.
    g = prog.NewContinuousVariables(nE, 'g')
    g_limits = prog.AddBoundingBoxConstraint(zeroE, onesE, g).evaluator()

    # Containers for the constraints.
    nV = len(gcs.Vertices())
    conservation_f = [None] * nV
    conservation_g = [None] * nV
    degree = [None] * nV

    for i, w in enumerate(gcs.Vertices()):
    
        # Conservation of flow for f.
        Ew_in = inedges_w(w)
        Ew_out = outedges_w(w)
        Ew = Ew_in + Ew_out
        fw = f[Ew]
        A = np.hstack((np.ones((1, len(Ew_in))), - np.ones((1, len(Ew_out)))))
        if w == s:
            conservation_f[i] = prog.AddLinearEqualityConstraint(A, [-1], fw).evaluator()
        else:
            conservation_f[i] = prog.AddLinearEqualityConstraint(A, [0], fw).evaluator()

        # Conservation of flow for g.
        gw = g[Ew]
        if w == t:
            conservation_g[i] = prog.AddLinearEqualityConstraint(A, [1], gw).evaluator()
        else:
            conservation_g[i] = prog.AddLinearEqualityConstraint(A, [0], gw).evaluator()

        # Degree constraints (redundant if indegree of w is 0).
        if len(Ew_in) > 0:
            A = np.ones((1, 2 * len(Ew_in)))
            fgin = np.concatenate((f[Ew_in], g[Ew_in]))
            degree[i] = prog.AddLinearConstraint(A, [0], [1], fgin).evaluator()

    redundant_edges = []
    for e in gcs.Edges():

        i = gcs.Vertices().index(e.u())
        j = gcs.Vertices().index(e.v())

        # Update bounds of consevation of flow.
        if s == e.u():
            f_limits.set_bounds(zeroE, zeroE)
            conservation_f[i].set_bounds([0], [0])
        else:
            conservation_f[i].set_bounds([1], [1])
        if t == e.v():
            g_limits.set_bounds(zeroE, zeroE)
            conservation_g[j].set_bounds([0], [0])
        else:
            conservation_g[j].set_bounds([-1], [-1])

        # Update bounds of degree constraints.
        degree[j].set_bounds([0], [0])

        # Check if edge e = (u,v) is redundant. 
        result = Solve(prog)
        if not result.is_success():
            redundant_edges.append(e)

        # Reset constraint bounds.
        if s == e.u():
            f_limits.set_bounds(zeroE, onesE)
            conservation_f[i].set_bounds([-1], [-1])
        else:
            conservation_f[i].set_bounds([0], [0])
        if t == e.v():
            g_limits.set_bounds(zeroE, onesE)
            conservation_g[j].set_bounds([1], [1])
        else:
            conservation_g[j].set_bounds([0], [0])
        degree[j].set_bounds([0], [1])

    # Remove redundant edges and isolated vertices.
    for e in redundant_edges:
        gcs.RemoveEdge(e)
    removeIsolatedVertices()

    if verbose:
        print('Vertices after preprocessing:', len(gcs.Vertices()))
        print('Edges after preprocessing:', len(gcs.Edges()))

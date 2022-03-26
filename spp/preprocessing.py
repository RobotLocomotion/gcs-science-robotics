import numpy as np
from pydrake.all import MathematicalProgram, Solve, ge
from copy import copy
from numbers import Number

def removeRedundancies(gcs, s, t, tol=1e-4, verbose=False):

    if verbose:
        print('Vertices before preprocessing:', len(gcs.Vertices()))
        print('Edges before preprocessing:', len(gcs.Edges()))
    
    for e in copy(gcs.Edges()):

        prog = MathematicalProgram()
        
        # Flow variables from s to u.
        nE = len(gcs.Edges())
        if s != e.u():
            f = prog.NewContinuousVariables(nE, 'f')
            fs = prog.NewContinuousVariables(1)[0]
            prog.AddLinearConstraint(ge(f, 0))
            prog.AddLinearConstraint(fs >= 0)
        else:
            f = np.zeros(nE)
            fs = 1
        prog.AddLinearCost(- fs)

        # Flow variables from v to t.
        if t != e.v():
            g = prog.NewContinuousVariables(nE, 'g')
            gv = prog.NewContinuousVariables(1)[0]
            prog.AddLinearConstraint(ge(g, 0))
            prog.AddLinearConstraint(gv >= 0)
        else:
            g = np.zeros(nE)
            gv = 1
        prog.AddLinearCost(- gv)
        
        for w in gcs.Vertices():
            
            # Conservation of flow for f.
            Ein = [k for k, e in enumerate(gcs.Edges()) if e.v() == w]
            Eout = [k for k, e in enumerate(gcs.Edges()) if e.u() == w]
            fin = sum(f[k] for k in Ein)
            fout = sum(f[k] for k in Eout)
            diff_f = fout - fin
            if w == s:
                diff_f -= fs
            if w == e.u():
                diff_f += fs
            if not isinstance(diff_f, Number):
                prog.AddLinearConstraint(diff_f == 0)
            
            # Conservation of flow for g.
            gin = sum(g[k] for k in Ein)
            gout = sum(g[k] for k in Eout)
            diff_g = gout - gin
            if w == e.v():
                diff_g -= gv
            if w == t:
                diff_g += gv
            if not isinstance(diff_g, Number):
                prog.AddLinearConstraint(diff_g == 0)
            
            # Degree constraint for cumulative f + g.
            if w == s or w == e.v():
                fg_w = fout + gout
            else:
                fg_w = fin + gin
            if not isinstance(fg_w, Number):
                prog.AddLinearConstraint(fg_w <= 1)
            
        # Check if edge e = (u,v) is redundant. 
        result = Solve(prog)
        if not result.is_success() or result.get_optimal_cost() > - 2 + tol:
            gcs.RemoveEdge(e)
            
    # Remove isolated vertices.
    for w in copy(gcs.Vertices()):
        E = [e for e in gcs.Edges() if e.u() == w or e.v() == w]
        if len(E) == 0:
            gcs.RemoveVertex(w)

    if verbose:
        print('Vertices after preprocessing:', len(gcs.Vertices()))
        print('Edges after preprocessing:', len(gcs.Edges()))

import torch
import numpy as np
from scipy.stats import rankdata
import MDAnalysis as mda

from sinatra_pro.mesh import Mesh

def reconstruct_by_sorted_threshold(
    mesh: Mesh, 
    directions: np.ndarray, 
    rates: torch.Tensor, 
    n_filtrations: int = 25, 
    n_directions_per_cone: int = 1, 
    ball_radius: float = 1.0, 
    by_rank: bool = False
) -> np.ndarray:
    n_directions = directions.shape[0]
    n_cones = int(n_directions / n_directions_per_cone)
    n_vertices = mesh.n_vertices
    # print(f"n_directions: {n_directions}, n_cones: {n_cones}, n_directions_per_cone: {n_directions_per_cone}")
    # print(f"rates.shape: {rates.shape}")
    rates_vert = np.zeros((n_vertices, n_cones, n_directions_per_cone),dtype=float)
    for i in range(n_cones):
        for j in range(n_directions_per_cone):
            k = i * n_directions_per_cone + j
            vertex_function = np.dot(mesh.vertices, directions[k])
            radius = np.linspace(-ball_radius,ball_radius,n_filtrations)
            filtration = np.digitize(vertex_function, radius)-1
            rates_vert[:,i,j] = rates[k * n_filtrations + filtration]
    height = np.amax(np.amin(rates_vert[:,:,:],axis=2),axis=1)
    if by_rank:
        rank = rankdata(height,method='dense')
        rank = rank/np.amax(rank)
        return rank
    else:
        return height


def project_rate_on_nonvacuum(rates,not_vacuum):
    rates_new = np.zeros(not_vacuum.size,dtype=float)
    j = 0
    for i in range(not_vacuum.size):
        if not_vacuum[i]:
            rates_new[i] = rates[j]
            j += 1
    return rates_new


def write_vert_prob_on_pdb(
    vert_prob: np.ndarray, 
    pdb_in_file: str, 
    pdb_out_file: str,
    selection: str = "protein",
    by_rank: bool = True
):
    u = mda.Universe(pdb_in_file)
    _protein = u.select_atoms(selection)
    _protein.write('.tmp.pdb')
    u_protein = mda.Universe(".tmp.pdb")
    u_protein.add_TopologyAttr('tempfactors')
    protein = u_protein.select_atoms(selection)
    if by_rank:
        y = rankdata(vert_prob,method='dense').astype(float)
        y *= 100.0/np.amax(y)
    else:
        ymin = np.amin(vert_prob)
        ymax = np.amax(vert_prob)
        y = (vert_prob - ymin)/(ymax-ymin)*100
    protein.tempfactors = y
    protein.write(pdb_out_file)
    return

# def write_vert_prob_on_pdb_residue(vert_prob,protA=None,protB=None,selection="protein",pdb_in_file=None,pdb_out_file=None,by_rank=True):
#     import MDAnalysis as mda
#     if selection == None:
#         selection = "protein"
#     if pdb_in_file == None:
#         pdb_in_file = "%s_%s/pdb/%s/%s_frame0.pdb"%(protA,protB,protA,protA)
#     if pdb_out_file == None:
#         pdb_out_file = "%s_%s/%s_reconstructed.pdb"%(protA,protB,protA)
#     u = mda.Universe(pdb_in_file)
#     protein = u.select_atoms(selection)
#     u.add_TopologyAttr('tempfactors') 
#     n_atom = len(protein)    
#     y = np.zeros(n_atom,dtype=float)
#     ag_res = u.atoms.groupby('resids')
#     rate_res = np.zeros(len(ag_res),dtype=float)
#     for i_r, res in enumerate(ag_res):
#         rate = 0
#         for a in ag_res[res]:
#             rate += vert_prob[a.ix]
#         rate /= len(ag_res[res])
#         rate_res[i_r] = rate
#     if by_rank:
#         rank_res = rankdata(rate_res,method='dense').astype(float)
#         rank_res *= 100.0/np.amax(rank_res)
#         for i_r, res in enumerate(ag_res):
#             for a in ag_res[res]:
#                 y[a.ix] = rank_res[i_r]
#     else:
#         for i_r, res in enumerate(ag_res):
#             for a in ag_res[res]:
#                 y[a.ix] = rate_res[i_r]
#         ymin = np.amin(y)
#         ymax = np.amax(y)
#         y = (y - ymin)/(ymax-ymin)*100
#     protein.tempfactors = y
#     protein.write(pdb_out_file)
#     return




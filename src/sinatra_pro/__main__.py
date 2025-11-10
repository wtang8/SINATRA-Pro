import torch
import os
from tqdm import trange

from sinatra_pro.atomic import convert_pdb_to_meshes
from sinatra_pro.mesh import Mesh
from sinatra_pro.directions import generate_equidistributed_cones
from sinatra_pro.euler import compute_ec_curve
from sinatra_pro.rate import calc_rate
from sinatra_pro.reconstruction import reconstruct_by_sorted_threshold, write_vert_prob_on_pdb

def __main__(
    dir_out: str, ### output directory
    dir_prot_A: str,
    dir_prot_B: str,
    align_selection: str = "protein and name CA",
    mesh_selection: str = "protein and not type H",
    align_sequence: bool = True,
    radius_sim: float = 3.0, # radius for simplicial construction
    hemisphere: bool = True, # distribute directions over hemisphere instead of whole sphere
    ec_type: str = 'DECT', # type of Euler characteristic measure (DECT/ECT/SECT), default: DECT', default='DECT'
    n_cones: int = 20, # number of cone
    n_direction_per_cone: int = 6, # number of direction per cone
    cap_radius: float = 0.2, # cap radius for each cone
    n_filtrations: int = 60, # number of filtration step
    bandwidth: float = 0.01, # bandwidth for elliptical slice sampling
    n_mcmc_steps: int = 2000, # number of sample from ESS
    n_burn_in_steps: int = 1000, # number of burn in steps for MCMC
    probit: bool = False, # use logistic likelihood instead of probit likelihood
    low_rank: bool = False, # use low rank matrix approximations to compute the RATE values
    verbose: bool = True
):
    
    os.makedirs(dir_out, exist_ok=True)

    ### Read PDB files for coordinates and Convert PDB to meshes
    print("Converting PDB to meshes...")
    meshes, labels = convert_pdb_to_meshes(
        dir_prot_A = dir_prot_A,
        dir_prot_B = dir_prot_B,
        align_selection = align_selection,
        mesh_selection = mesh_selection,
        radius_sim = radius_sim,
        align_sequence = align_sequence,
        verbose = verbose
    )
    n_meshes = len(meshes)
    labels = torch.tensor(labels, dtype=torch.float32)

    ## Calculate distributed cones of directions for EC calculations
    print("Generating directions for EC calculations")
    directions = generate_equidistributed_cones(
        n_cones=n_cones,
        n_direction_per_cone=n_direction_per_cone,
        cap_radius=cap_radius,
        hemisphere=hemisphere
    )
    n_directions = directions.shape[0]
    n_features = n_directions * n_filtrations

    ec_curves_meshes = torch.zeros((n_meshes, n_features), dtype=torch.float32)
    for i in trange(len(meshes), desc="Calculating EC for Meshes"):
        filtration_radius, ec_curves = compute_ec_curve(
            meshes[i], directions, n_filtrations, ball_radius = 1.0, ec_type = ec_type, include_faces = True
        )
        ec_curves_meshes[i, :] = torch.from_numpy(ec_curves.flatten())
    
    ec_curves_meshes_transposed = ec_curves_meshes.T # (n_features, n_samples)
    good_features = (ec_curves_meshes_transposed != 0.0).to(dtype=torch.float32).sum(dim=1) > 0
    
    ec_curves_meshes_transposed_masked = ec_curves_meshes_transposed[good_features, :]
    ec_curves_meshes_transposed_masked -= ec_curves_meshes_transposed_masked.mean(dim=1, keepdim=True)
    ec_curves_meshes_transposed_masked /= ec_curves_meshes_transposed_masked.std(dim=1, keepdim=True)

    ## RATE calculation for variable selections from the topological summary statistics
    result_rate = calc_rate(
        X = ec_curves_meshes_transposed_masked,
        y = labels,
        bandwidth = bandwidth,
        n_mcmc_steps = n_mcmc_steps,
        n_burn_in_steps = n_burn_in_steps,
        probit = probit,
        low_rank = low_rank,
        verbose = verbose
    )
    # kld = result_rate['KLD']
    rates = result_rate['RATE']
    # delta = result_rate['Delta']
    # eff_samp_size = result_rate['ESS']
    rates_unmasked = torch.zeros((n_features), dtype=torch.float32)
    rates_unmasked[good_features] = rates

    # for mesh in meshes:
    ## reconstruct the RATE values onto the protein structures for visualization
    ## reconstruct probabilities are stored in "Temperature factor" column in the pdb format
    ## can then be visualized using Chimera or Pymol  
    vert_prob = reconstruct_by_sorted_threshold(
        meshes[0], 
        directions,
        rates_unmasked,
        n_filtrations, 
        n_direction_per_cone,
    )

    print(f"vert_prob.shape: {vert_prob.shape}")
    

    from pathlib import Path 
    pdb_files_A_map = sorted(Path(dir_prot_A).glob("*.pdb"))[0]
    write_vert_prob_on_pdb(
        vert_prob = vert_prob, 
        pdb_in_file = str(pdb_files_A_map),
        pdb_out_file = os.path.join(dir_out, f"{Path(pdb_files_A_map).stem}_reco.pdb"),
        selection = mesh_selection,
        by_rank = True
    )
    
if __name__ == "__main__":
    import fire
    fire.Fire(__main__)
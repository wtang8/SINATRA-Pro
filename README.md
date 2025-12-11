# SINATRA Pro 

Protein Structure and Conformation Analysis using Topological Summary Statistics.

## Introduction

The sub-image selection problem is to identify physical regions that most explain the variation between two classes of three dimensional shapes. [SINATRA](https://github.com/lcrawlab/SINATRA) is a statistical pipeline for carrying out sub-image analyses using topological summary statistics (Wang et al. 2021). SINATRA Pro is an adaptation of the SINATRA framework for structure-based applications in protein dynamics. The general algorithm follows four key steps:

1. 3D shapes of protein structures (represented as triangular meshes) are summarized by a collection of vectors (or curves) detailing their topology (e.g., Euler characteristics, persistence diagrams, etc).
2. A statistical model is used to classify the shapes based on their topological summaries. Here, we make use of a Gaussian process classification model with a probit link function.
3. After fitting the model, an association measure is computed for each topological feature (e.g., centrality measures, posterior inclusion probabilities, p-values, etc).
4. Association measures are mapped back onto the original protein structures via a reconstruction algorithm, thus, highlighting atomic or residue-level positions that best explain the variation between two ensembles.

Through detailed simulations, we assess the power of our algorithm as a function of its free parameters. As an application of our pipeline, we conduct feature selection for identifying minute conformational changes in five independent protein systems of varying complexities.

## Package Details

Code for implementing the SINATRA Pro pipeline was written in Python 3 (version 3.8+). As part of this procedure:

1. Reading of trajectory files, alignment of protein structures are done using the [MDAnalysis](https://www.mdanalysis.org) package (Gowers et. al. 2016, Michaud-Agrawal et. al. 2011).
2. Neighbor search algorithms use KD-tree spatial indexing via [SciPy](https://www.scipy.org) for O(N log N) performance.
3. Mesh vertices are represented as [PyTorch](https://pytorch.org) tensors for efficient computation.
4. Most mathematical calculations are performed using [NumPy](https://numpy.org) and [SciPy](https://www.scipy.org).
5. Inference for the Gaussian process classification (GPC) model was done using elliptical slice sampling (Murray, Prescott, and MacKay 2010).
6. Association measures are computed for the Euler characteristic curves using the relative centrality criterion (RATE), which is a variable selection measure for nonlinear and nonparametric statistical methods (see Crawford et al. 2019 and Ish-Horowicz et al. 2019).

### Recent Improvements

The package has been recently refactored for improved performance and code quality:

- **KD-tree Optimization**: Neighbor list creation now uses KD-tree spatial indexing (O(N log N)) instead of distance matrices (O(N²)), providing significant speedup for large protein structures.
- **Comprehensive Test Suite**: 73+ unit tests across core modules (mesh, directions, euler) ensure code reliability and correctness.
- **Modern Type Hints**: All functions include type annotations for better IDE support and code clarity.
- **Enhanced Documentation**: NumPy-style docstrings with detailed parameter descriptions and examples throughout the codebase.

## Dependencies

The SINATRA Pro package requires Python 3.8 or higher and depends on the following packages:

    numpy >= 1.20.0
    scipy >= 1.7.0
    torch >= 1.10.0
    mdanalysis >= 2.0.0
    fast-histogram >= 0.11
    joblib >= 1.0.0

### Development Dependencies

For running tests and development:

    pytest >= 7.0.0
    pytest-cov >= 3.0.0

## Python Package Installation

To install the package:

```bash
pip install SINATRA-Pro
```

For development installation with test dependencies:

```bash
pip install SINATRA-Pro[dev]
```

To load the package:

```python
import sinatra_pro
from sinatra_pro.mesh import Mesh
from sinatra_pro.euler import compute_ec_curve
from sinatra_pro.directions import generate_equidistributed_cones
```

### Running Tests

After installation with dev dependencies:

```bash
cd python_package
pytest test/
```

### Programmatic Usage

Example of using the core API:

```python
import torch
import numpy as np
from sinatra_pro.mesh import Mesh
from sinatra_pro.directions import generate_equidistributed_cones
from sinatra_pro.euler import compute_ec_curve

# Create a mesh from vertices (e.g., protein CA atoms)
vertices = torch.rand(100, 3)  # 100 random 3D points
mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

# Normalize the mesh to unit sphere
mesh.normalize()

# Generate equidistributed directions for EC curves
directions = generate_equidistributed_cones(
    n_cones=50,
    cap_radius=0.8,
    n_direction_per_cone=1,
    hemisphere=False
)

# Compute Euler Characteristic curves
radius, ec_curves = compute_ec_curve(
    mesh,
    directions,
    n_filtrations=25,
    ball_radius=1.0,
    ec_type="DECT",
    include_faces=True
)

print(f"EC curves shape: {ec_curves.shape}")  # (50, 25)
```

To run the command-line application:

```bash
sinatra-pro
# or
python3 -m sinatra_pro
```

        usage: __main__.py [-h] [-pa PROTA] [-pb PROTB] [-sa STRUCT_FILE_A]
                           [-ta TRAJ_FILE_A] [-sb STRUCT_FILE_B] [-tb TRAJ_FILE_B]
                           [-dir DIRECTORY] [-pl] [-nc N_CORE] [-n N_SAMPLE]
                           [-of OFFSET] [-s SELECTION] [-r RADIUS] [-hs] [-et EC_TYPE]
                           [-c N_CONE] [-d N_DIRECTION_PER_CONE] [-t CAP_RADIUS]
                           [-l N_FILTRATION] [-bw BANDWIDTH] [-sm SAMPLING_METHOD]
                           [-nm N_MCMC_STEPS] [-ll] [-v] [-no]

        optional arguments:

              -h, --help            show this help message and exit
              -pa PROTA, --protA PROTA
                                    name of protein A for file naming
              -pb PROTB, --protB PROTB
                                    name of protein B for file naming
              -sa STRUCT_FILE_A, --struct_file_A STRUCT_FILE_A
                                    structure file for protein A (.gro)
              -ta TRAJ_FILE_A, --traj_file_A TRAJ_FILE_A
                                    trajectory file for protein A (.xtc)
              -sb STRUCT_FILE_B, --struct_file_B STRUCT_FILE_B
                                    structure file for protein B (.gro)
              -tb TRAJ_FILE_B, --traj_file_B TRAJ_FILE_B
                                    trajectory file for protein B (.xtc)
              -dir DIRECTORY, --directory DIRECTORY
                                    directory for output files
              -fp, --from_pdb       start from sets of PDB structures instead of
                                        trajectories
              -pa PDBPATH_A, --pdbpath_A PDBPATH_A
                                    directory containing PDB structures for protein A
              -pb PDBPATH_B, --pdbpath_B PDBPATH_B
                                    directory containing PDB structures for protein B
              -pr PDB_REFERENCE, --pdb_reference PDB_REFERENCE
                                    PDB structure for visualization from protein A
              -pl, --parallel
                                    use multiple CPU cores for calculations
              -nc N_CORE, --n_core N_CORE
                                    number of core for parallel computing, default: use
                                    all cores
              -n N_SAMPLE, --n_sample N_SAMPLE
                                    number of sample drawn from trajectory, default: 10
              -of OFFSET, --offset OFFSET
                                    starting frame for sample drawn from trajectory,
                                    default: 0
              -s SELECTION, --selection SELECTION
                                    selection for protein, default: all protein
              -r RADIUS, --radius RADIUS
                                    radius for simplicial construction, default: 2.0
              -hs, --hemisphere     distribute directions over hemisphere instead of whole
                                    sphere
              -et EC_TYPE, --ec_type EC_TYPE
                                    type of Euler characteristic measure (DECT/ECT/SECT),
                                    default: DECT
              -c N_CONE, --n_cones N_CONE
                                    number of cone, default: 1
              -d N_DIRECTION_PER_CONE, --n_direction_per_cone N_DIRECTION_PER_CONE
                                    number of direction per cone, default: 1
              -t CAP_RADIUS, --cap_radius CAP_RADIUS
                                    cap radius, default: 0.8
              -l N_FILTRATION, --n_filtrations N_FILTRATION
                                    number of filtration step, default: 20
              -bw BANDWIDTH, --bandwidth BANDWIDTH
                                    bandwidth for elliptical slice sampling, default: 0.01
              -sm SAMPLING_METHOD, --sampling_method SAMPLING_METHOD
                                    sampling method, default: ESS
              -nm N_MCMC_STEPS, --n_mcmc_steps N_MCMC_STEPS
                                    number of sample from ESS
              -ll, --logistic_likelihood
                                    use logistic likelihood instead of probit likelihood
              -lr, --low_rank       use low rank matrix approximations to compute the RATE
                                    values
              -v, --verbose         verbose
              -no, --name_offset    name folder with offset

## Examples of Running the Package

Starting from MD trajectories

        python3 -m sinatra_pro --protA WT --protB R164S \
                --directory "WT_R164S_65_213_no164sc_2.0" \
                --n_sample 10 \
                --struct_file_A "WT.gro" \
                --traj_file_A "WT.xtc" \
                --struct_file_B "R164S.gro" \
                --traj_file_B "R164S.xtc" \
                --selection "protein and resid 65:213 and not (resid 164 and not backbone)" \
                --radius 2.0 \
                --n_cones 4 \
                --n_direction_per_cone 4 \
                --cap_radius 0.80 \
                --ec_type "DECT" \
                --n_filtrations 60 \
                --n_mcmc_steps 100000 \
                --parallel \
                --n_core 4 --verbose

Starting from aligned PDB structures

        python3 -m sinatra_pro --protA WT --protB R164S \
                --directory "WT_R164S_65_230_2.0" \
                --n_sample 10 \
                --from_pdb \
                --pdbpath_A "WT_R164S_65_230_2.0/pdb/WT_offset_0/" \
                --pdbpath_B "WT_R164S_65_230_2.0/pdb/R164S_offset_0/" \
                --pdb_reference "WT_R164S_65_230_2.0/pdb/WT_offset_0/WT_frame0.pdb" \
                --radius 2.0 \
                --n_cones 1 \
                --n_direction_per_cone 1 \
                --cap_radius 0.80 \
                --ec_type "DECT" \
                --n_filtrations 20 \
                --n_mcmc_steps 10000 \
                --parallel \
                --n_core 4 --verbose

Other code specific to analyses conducted in the paper can be found in the repo [SINATRA_Pro_Paper_Results](https://github.com/lcrawlab/SINATRA_Pro_Paper_Results).

## Questions and Feedback

For questions or concerns, please contact [Wai Shing Tang](mailto:wai_shing_tang@brown.edu) or [Lorin Crawford](mailto:lcrawford@microsoft.com). We appreciate any feedback you may have with our repository and instructions for running the software.

## Relevant Citations

Wai Shing Tang*, Gabriel Monteiro da Silva*, Henry Kirveslahti, Erin Skeens, Bibo Feng, Timothy Sudijono, Kevin K. Yang, Sayan Mukherjee, Brenda Rubenstein, and Lorin Crawford. Topological data analytic approach for discovering biophysical signatures in protein dynamics. _bioRxiv_.

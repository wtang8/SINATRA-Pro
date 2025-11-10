"""
Atomic-level PDB to Mesh Conversion Module.

This module provides simplified functions for converting PDB files directly
to Mesh objects with automatic alignment and labeling.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import MDAnalysis as mda
from MDAnalysis.analysis import align
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from MDAnalysis.lib.util import convert_aa_code
from MDAnalysis.core.groups import AtomGroup
from sinatra_pro.mesh import Mesh


# --- Extract sequences ---
def _extract_sequence(universe, selection="protein"):
    residues = universe.select_atoms(selection).residues
    seq = "".join(
        [convert_aa_code(res.resname) for res in residues if convert_aa_code(res.resname)]
    )
    return seq, residues  # return residues to map indices later

def _get_sequence_alignment_masks(
    pdb_file_A: str,
    pdb_file_B: str
) -> Tuple[List[bool], List[bool]]:
    """
    Compute sequence alignment between two structures using Needleman-Wunsch.

    Parameters
    ----------
    pdb_file_A : str
        PDB file path for protein A
    pdb_file_B : str
        PDB file path for protein B

    Returns
    -------
    seqsel_A : List[bool]
        Boolean mask for protein A residues
    seqsel_B : List[bool]
        Boolean mask for protein B residues
    """

    # Load the two structures
    u_A = mda.Universe(pdb_file_A)
    u_B = mda.Universe(pdb_file_B)

    seq_A, residues_A = _extract_sequence(u_A)
    seq_B, residues_B = _extract_sequence(u_B)

    # --- Align sequences using Biopython ---
    alignments = pairwise2.align.globalxx(seq_A, seq_B)
    print(format_alignment(*alignments[0]))

    # Take the *best* alignment result
    seqA_aligned, seqB_aligned, score, begin, end = alignments[0]
    n_res = len(seqA_aligned)

    # --- Map aligned residues back to MDAnalysis selections ---
    seqsel_A, seqsel_B = [], []
    idx_A, idx_B = 0, 0  # residue indices within original sequences

    for i in range(n_res):
        if seqA_aligned[i] != "-" and seqB_aligned[i] != "-":
            seqsel_A.append(residues_A[idx_A])
            seqsel_B.append(residues_B[idx_B])
            idx_A += 1
            idx_B += 1
        elif seqA_aligned[i] == "-" and seqB_aligned[i] != "-":
            idx_B += 1
        elif seqB_aligned[i] == "-" and seqA_aligned[i] != "-":
            idx_A += 1

    # # Convert to MDAnalysis AtomGroups (useful for RMSD / structural alignment)
    # atoms_A = u_A.select_atoms("resid " + " ".join(str(r.resid) for r in seqsel_A))
    # atoms_B = u_B.select_atoms("resid " + " ".join(str(r.resid) for r in seqsel_B))

    # print(f"Matched residues: {len(seqsel_A)}")
    # print("Atoms A:", atoms_A)
    # print("Atoms B:", atoms_B)

    return seqsel_A, seqsel_B


def convert_pdb_to_meshes(
    dir_prot_A: str,
    dir_prot_B: str,
    align_selection: str = "protein and name CA",
    mesh_selection: str = "protein and not type H",
    radius_sim: float = 3.0,
    align_sequence: bool = False,
    verbose: bool = True
) -> Tuple[List[Mesh], List[int]]:
    """
    Convert PDB files from two directories to aligned meshes.

    Reads PDB files from two directories, aligns all structures to the first
    frame (from protein A), and generates simplicial meshes. Optionally performs
    sequence alignment to handle proteins with different sequences.

    Parameters
    ----------
    dir_prot_A : str
        Directory containing PDB files for protein A
    dir_prot_B : str
        Directory containing PDB files for protein B
    selection : str, default="name CA"
        MDAnalysis selection string (e.g., "name CA", "protein and not type H")
    radius_sim : float, default=3.0
        Radius cutoff for simplicial mesh construction (in Angstroms)
    align_sequence : bool, default=False
        If True, perform Needleman-Wunsch sequence alignment between proteins
        A and B to handle different sequences (e.g., mutations, indels)
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    meshes : List[Mesh]
        List of Mesh objects for all structures (A followed by B)
    labels : List[int]
        List of labels: -1 for protein A, 1 for protein B

    Examples
    --------
    >>> # Basic usage without sequence alignment
    >>> meshes, labels = convert_pdb_to_meshes(
    ...     dir_prot_A="data/WT/",
    ...     dir_prot_B="data/WT_replica/",
    ...     selection="name CA",
    ...     radius_sim=2.0
    ... )

    >>> # With sequence alignment for mutants
    >>> meshes, labels = convert_pdb_to_meshes(
    ...     dir_prot_A="data/WT/",
    ...     dir_prot_B="data/R164S/",
    ...     selection="protein",
    ...     radius_sim=2.0,
    ...     align_sequence=True
    ... )
    """
    # Convert to Path objects
    path_A = Path(dir_prot_A)
    path_B = Path(dir_prot_B)

    # Get all PDB files, sorted alphabetically
    pdb_files_A = sorted(path_A.glob("*.pdb"))
    pdb_files_B = sorted(path_B.glob("*.pdb"))

    if len(pdb_files_A) == 0:
        raise ValueError(f"No PDB files found in {dir_prot_A}")
    if len(pdb_files_B) == 0:
        raise ValueError(f"No PDB files found in {dir_prot_B}")

    if verbose:
        print(f"Found {len(pdb_files_A)} PDB files in {dir_prot_A}")
        print(f"Found {len(pdb_files_B)} PDB files in {dir_prot_B}")

    # Perform sequence alignment if requested
    seqsel_A: Optional[List[bool]] = None
    seqsel_B: Optional[List[bool]] = None

    if align_sequence:
        if verbose:
            print("\nPerforming sequence alignment between protein A and B...")

        seqsel_A, seqsel_B = _get_sequence_alignment_masks(
            str(pdb_files_A[0]),
            str(pdb_files_B[0])
        )

        if verbose:
            n_aligned = sum(seqsel_A)
            print(f"Sequence alignment complete:")
            print(f"  Protein A: {len(seqsel_A)} residues, {n_aligned} aligned")
            print(f"  Protein B: {len(seqsel_B)} residues, {n_aligned} aligned")

    meshes: List[Mesh] = []
    labels: List[int] = []
    max_radius: float = 0.0

    # Reference structure: first frame from protein A
    frame_ref: AtomGroup | None = None
    frame_ref_CA: AtomGroup | None = None

    # Process protein A files
    for i, pdb_file in enumerate(pdb_files_A):
        if verbose:
            sys.stdout.write(
                f"Processing protein A: {i+1}/{len(pdb_files_A)} "
                f"({pdb_file.name})\r"
            )
            sys.stdout.flush()

        # Load structure
        u = mda.Universe(str(pdb_file))
        atoms_mesh = u.select_atoms(mesh_selection)

        _align_selection = align_selection
        # Apply sequence alignment mask if needed
        if align_sequence and seqsel_A is not None:
            _align_selection = "resid " + " ".join(str(r.resid) for r in seqsel_A) + " and " + align_selection
        
        # Set reference frame (first file)
        if frame_ref is None:
            atoms_align = u.select_atoms(align_selection)
            if len(atoms_align) == 0:
                raise ValueError(
                    f"Alignment selection '{atoms_align}' returned no atoms in {pdb_file}"
                )
            frame_ref = atoms_align
            frame_ref_CA = atoms_align.select_atoms(_align_selection)

            if frame_ref_CA is None or len(frame_ref_CA) == 0:
                raise ValueError(
                    f"No CA atoms found in reference structure {pdb_file}"
                )

            # Center reference at origin
            frame_ref.translate(-frame_ref_CA.center_of_mass())

            if verbose:
                print(f"\nUsing {pdb_file.name} as reference frame")
                print(f"Reference has {len(frame_ref)} atoms")
                print(f"Using {len(frame_ref_CA)} atoms for alignment")
        else:
            # Align to reference
            try:
                align.alignto(atoms_mesh, frame_ref, select=_align_selection, weights=None)
            except Exception as e:
                raise ValueError(
                    f"Failed to align {pdb_file.name} to reference: {e}"
                )

        # Generate mesh
        vertices = torch.from_numpy(atoms_mesh.positions.copy()).float()
        mesh = Mesh(vertices, generate_mesh=True, radius=radius_sim)
        radius = mesh.calc_radius()
        max_radius = max(max_radius, radius)

        meshes.append(mesh)
        labels.append(-1)

    if verbose:
        print(f"\nConverted {len(pdb_files_A)} frames of protein A into meshes")

    # Process protein B files
    for i, pdb_file in enumerate(pdb_files_B):
        if verbose:
            sys.stdout.write(
                f"Processing protein B: {i+1}/{len(pdb_files_B)} "
                f"({pdb_file.name})\r"
            )
            sys.stdout.flush()

        # Load structure
        u = mda.Universe(str(pdb_file))

        _align_selection = align_selection
        # Apply sequence alignment mask if needed
        if align_sequence and seqsel_B is not None:
            _align_selection = "resid " + " ".join(str(r.resid) for r in seqsel_B) + " and " + align_selection
        atoms_mesh = u.select_atoms(mesh_selection)
        if len(atoms_mesh) == 0:
            raise ValueError(
                f"Mesh selection '{mesh_selection}' returned no atoms in {pdb_file}"
            )

        # Align to reference
        try:
            align.alignto(atoms_mesh, frame_ref, select=_align_selection, weights=None)
        except Exception as e:
            raise ValueError(
                f"Failed to align {pdb_file.name} to reference: {e}"
            )

        # Generate mesh
        vertices = torch.from_numpy(atoms_mesh.positions.copy()).float()
        mesh = Mesh(vertices, generate_mesh=True, radius=radius_sim)
        radius = mesh.calc_radius()
        max_radius = max(max_radius, radius)

        meshes.append(mesh)
        labels.append(1)

    if verbose:
        print(f"\nConverted {len(pdb_files_B)} frames of protein B into meshes")
        print(f"Maximum radius: {max_radius:.3f} Å")
        print("Normalizing meshes...")

    # Normalize all meshes
    for mesh in meshes:
        # Scale vertices to unit sphere
        mesh.normalize(max_radius)

    if verbose:
        print(f"\nNormalization complete: all meshes scaled to unit sphere")
        print(f"\nTotal: {len(meshes)} meshes generated")
        print(f"Labels: {labels.count(-1)} × protein A, {labels.count(1)} × protein B")

    return meshes, labels


def load_single_pdb_to_mesh(
    pdb_file: str,
    selection: str = "name CA",
    radius_sim: float = 3.0,
    normalize: bool = False
) -> Mesh:
    """
    Load a single PDB file and convert to Mesh.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file
    selection : str, default="name CA"
        MDAnalysis selection string
    radius_sim : float, default=3.0
        Radius cutoff for simplicial mesh construction
    normalize : bool, default=False
        If True, normalize mesh to unit sphere

    Returns
    -------
    Mesh
        Mesh object generated from PDB structure

    Examples
    --------
    >>> mesh = load_single_pdb_to_mesh("protein.pdb", selection="name CA")
    >>> print(f"Mesh has {mesh.n_vertices} vertices")
    """
    pdb_path = Path(pdb_file)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    # Load structure
    u = mda.Universe(str(pdb_file))
    atoms = u.select_atoms(selection)

    if len(atoms) == 0:
        raise ValueError(
            f"Selection '{selection}' returned no atoms in {pdb_file}"
        )

    # Center at origin
    atoms.translate(-atoms.center_of_mass())

    # Generate mesh
    vertices = torch.from_numpy(atoms.positions.copy()).float()
    mesh = Mesh(vertices, generate_mesh=True, radius=radius_sim)

    # Normalize if requested
    if normalize:
        mesh.normalize()

    return mesh

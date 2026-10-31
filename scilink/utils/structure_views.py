"""Viewing angles for rendering an atomic structure.

Shared by the simulation agents' structure views and the web UI's file
preview. numpy only: the caller brings the ASE ``Atoms``.
"""

import numpy as np

# Plain orthogonal views: the default, and the only choice for a structure
# with no cell (a molecule from a bare .xyz), where the heuristics below
# have no cell lengths to read.
AXIS_VIEWS = {
    'x': '0y,90x,0z',
    'y': '-90x,0y,0z',
    'z': '0x,0y,0z'
}


def _center_structure_for_visualization(atoms):
    """Center structure in cell for better visualization"""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    
    # Center of mass
    center_of_mass = positions.mean(axis=0)
    
    # Cell center  
    cell_center = cell.sum(axis=0) / 2
    
    # Shift to center
    shift = cell_center - center_of_mass
    atoms.translate(shift)
    atoms.wrap()  # Keep atoms in cell
    
    return atoms

def _get_optimal_rotations(atoms):
    """Automatically determine optimal viewing angles with simple heuristics"""
    
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    
    # Quick checks using existing ASE functionality
    
    # 1. Check for slab structure (large vacuum gap)
    if _is_slab_structure(cell, positions):
        return _get_slab_rotations(cell, positions)
    
    # 2. Check for highly anisotropic cell
    elif _is_anisotropic(cell):
        return _get_anisotropic_rotations(cell)
    
    # 3. Check for layered structure
    elif _is_layered_structure(atoms):
        return _get_layered_rotations(cell)
    
    # 4. Default: use orthogonal views
    else:
        return dict(AXIS_VIEWS)

def _is_slab_structure(cell, positions):
    """Detect slab by checking for a >5 Å vacuum gap along any axis.

    Compares each cell length against the atomic extent along the SAME
    axis (not just z), so slabs with vacuum along a or b are detected
    correctly and anisotropic crystals without vacuum aren't false-
    positive."""
    cell_lengths = np.linalg.norm(cell, axis=1)
    extents = positions.max(axis=0) - positions.min(axis=0)
    return any(length > extent + 5.0 for length, extent in zip(cell_lengths, extents))

def _get_slab_rotations(cell, positions):
    """Views optimized for slab structures"""
    # The vacuum runs along the axis with the most empty space, the same
    # comparison _is_slab_structure makes. Not the longest cell vector: a
    # wide in-plane supercell (6x6 MoS2, ~19 A) outgrows a ~18 A cell height.
    lengths = np.linalg.norm(cell, axis=1)
    extents = positions.max(axis=0) - positions.min(axis=0)
    vacuum_dir = np.argmax(lengths - extents)
    
    if vacuum_dir == 2:  # vacuum along z
        return {
            'surface': '0x,0y,0z',      # Top view of surface
            'edge1': '90x,0y,0z',       # Edge view 1  
            'edge2': '0x,90y,0z'        # Edge view 2
        }
    elif vacuum_dir == 1:  # vacuum along y
        return {
            'surface': '90x,0y,0z',
            'edge1': '0x,0y,0z', 
            'edge2': '0x,0y,90z'
        }
    else:  # vacuum_dir == 0, vacuum along x
        return {
            'surface': '0x,90y,0z',
            'edge1': '0x,0y,0z',
            'edge2': '90x,0y,0z'
        }

def _is_anisotropic(cell):
    """Check if cell is highly anisotropic"""
    lengths = np.linalg.norm(cell, axis=1)
    ratio = lengths.max() / lengths.min()
    return ratio > 2.0  # If one direction is 2x larger than another

def _get_anisotropic_rotations(cell):
    """Views aligned with cell vectors for anisotropic structures"""
    # Find principal directions
    lengths = np.linalg.norm(cell, axis=1)
    
    # Sort by length to identify short/long directions
    sorted_indices = np.argsort(lengths)
    
    return {
        'along_short': f'0x,0y,{90 if sorted_indices[0] != 2 else 0}z',
        'along_medium': f'{90 if sorted_indices[1] == 0 else 0}x,{90 if sorted_indices[1] == 1 else 0}y,0z',
        'along_long': f'{90 if sorted_indices[2] == 0 else 0}x,{90 if sorted_indices[2] == 1 else 0}y,{90 if sorted_indices[2] == 2 else 0}z'
    }

def _is_layered_structure(atoms):
    """Simple check for layered materials based on z-coordinates"""
    positions = atoms.get_positions()
    z_coords = positions[:, 2]
    
    # Check if atoms form distinct layers (gaps > 2 Å between groups)
    sorted_z = np.sort(z_coords)
    gaps = np.diff(sorted_z)
    
    return np.any(gaps > 2.0)  # Significant gaps indicate layers

def _get_layered_rotations(cell):
    """Views for layered structures"""
    return {
        'layers': '90x,0y,0z',      # Side view to see layers
        'in_plane': '0x,0y,0z',     # Top view of layer plane
        'oblique': '45x,0y,45z'     # Oblique view for 3D perspective
    }

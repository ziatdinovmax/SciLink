"""X-ray mass / linear attenuation coefficients from spekpy (NIST tables).

Exposed to the hyperspectral code-gen sandbox as ``attenuation(...)`` so
generated scripts can obtain mu on the datacube's own energy axis instead of
the user supplying a mu-table operand. spekpy is a heavy, OPTIONAL dependency,
imported lazily inside the function — importing this module never pulls it in,
and a missing install raises a clear, actionable error.

Lives in ``_shared`` because X-ray attenuation is generic physics reusable
across X-ray techniques; a graduated X-ray skill can re-export it as a
``TOOL_SPEC`` for scoped visibility.
"""
import numpy as np

# Element symbol -> atomic number (Z), 1..92. spekpy keys elements by Z.
_SYMBOL_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
    'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33,
    'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41,
    'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49,
    'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
    'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73,
    'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81,
    'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89,
    'Th': 90, 'Pa': 91, 'U': 92,
}


def attenuation(material, energy_kev, density=None):
    """X-ray mass (cm^2/g) or linear (1/cm) attenuation coefficient vs energy.

    Generates mu(E) from NIST tables (via spekpy) on any energy grid, with K/L
    absorption edges at the correct energies automatically — replacing hand-built
    tables or user-supplied mu files.

    Parameters
    ----------
    material : str | int
        Element symbol ('Au', 'Pb'), atomic number (79), or a spekpy-registered
        material name.
    energy_kev : array-like
        Photon energies in keV (any grid, e.g. the datacube's energy axis).
    density : float | None
        If given (g/cm^3) -> returns LINEAR attenuation mu = (mu/rho)*density
        [1/cm], directly usable in Beer-Lambert  -ln(T) = mu * t. If None ->
        returns the MASS attenuation coefficient mu/rho [cm^2/g].

    Returns
    -------
    np.ndarray, same shape as energy_kev — mu/rho (cm^2/g) or mu (1/cm).

    Notes
    -----
    Energies below ~1 keV are clamped to the 1 keV table endpoint. Raises
    ImportError (with an install hint) if spekpy is absent, ValueError for an
    unknown element symbol / unregistered material.
    """
    try:
        from spekpy import SpekTools as _T  # lazy: heavy optional dependency
    except ImportError as e:
        raise ImportError(
            "attenuation() needs spekpy (optional). Install it with "
            "`pip install spekpy`."
        ) from e

    mu_data, _, _ = _T.load_mu_data('nist')
    E = np.clip(np.asarray(energy_kev, dtype=float), 1.0, None)

    if isinstance(material, (int, np.integer)):
        mor = mu_data.get_mu_over_rho(int(material), E)
    elif isinstance(material, str) and material in _SYMBOL_TO_Z:
        mor = mu_data.get_mu_over_rho(_SYMBOL_TO_Z[material], E)
    elif isinstance(material, str):
        try:
            mor = mu_data.get_mu_over_rho_composition(material, E)
        except Exception as e:
            raise ValueError(
                f"{material!r} is not an element symbol and not a spekpy-"
                f"registered material. Use an element symbol ('Au') or Z (79)."
            ) from e
    else:
        raise ValueError(f"Unrecognized material: {material!r}")

    mor = np.asarray(mor, dtype=float)
    return mor * float(density) if density is not None else mor


from ._spec import ToolSpec  # noqa: E402

TOOL_SPEC = ToolSpec(
    name="attenuation",
    description=(
        "X-ray mass (cm^2/g) or linear (1/cm) attenuation coefficient vs energy "
        "from NIST tables (spekpy). Returns mu on your energy axis for an element, "
        "with K/L absorption edges at the correct energies — so you can derive mu "
        "for Beer-Lambert / K-edge thickness from the element name instead of a "
        "supplied mu-table operand."
    ),
    signature="attenuation(material, energy_kev, density=None) -> np.ndarray",
    import_line="from scilink.skills._shared.xray_attenuation import attenuation",
    parameters={
        "material": "Element symbol ('Au','Pb'), atomic number (79), or a spekpy material name.",
        "energy_kev": "Energies in keV — pass the datacube's energy axis so mu is returned grid-aligned.",
        "density": "g/cm^3 -> LINEAR mu (1/cm) for -ln(T)=mu*t; omit -> MASS mu/rho (cm^2/g).",
    },
    required=["material", "energy_kev"],
    agents=["hyperspectral"],
    when_to_use=(
        "X-ray transmission / K-edge thickness work when you need mu(E) for a known "
        "element and no mu operand is supplied — derive it here from the element name."
    ),
    returns="np.ndarray of mu/rho (cm^2/g) or linear mu (1/cm), same shape as energy_kev.",
    example="mu = attenuation('Au', energy_kev)   # cm^2/g on the data's energy axis",
)


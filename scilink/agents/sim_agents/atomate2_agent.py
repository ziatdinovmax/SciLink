import os
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from atomate2.vasp.sets.base import VaspInputGenerator

class Atomate2InputAgent:
    """
    Wraps Atomate2's VaspInputGenerator to produce VASP inputs for SciLink workflows.
    """

    def __init__(self,
                 incar_settings: dict = None,
                 kpoints_settings: dict = None,
                 potcar_settings: dict = None):
        self.gen = VaspInputGenerator(
            user_incar_settings=incar_settings or {},
            user_kpoints_settings=kpoints_settings or {},
            user_potcar_settings=potcar_settings or {}
        )

    def generate(self, structure, output_dir: str) -> str:
        """
        Generate VASP input files in the specified directory.

        Args:
            structure:         A pymatgen Structure or an ASE Atoms object.
            output_dir (str):  Directory to write VASP input files.

        Returns:
            str: Path to the output directory containing the input files.
        """
        # 1) Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        # 2) Convert ASE Atoms → pymatgen Structure if needed
        if not isinstance(structure, Structure):
            try:
                structure = AseAtomsAdaptor.get_structure(structure)
            except Exception as e:
                raise TypeError(
                    "Atomate2InputAgent.generate: 'structure' must be a "
                    "pymatgen Structure or ASE Atoms"
                ) from e

        # 3) Build POSCAR/INCAR/KPOINTS via Atomate2, but skip POTCAR
        vis = self.gen.get_input_set(structure)
        os.makedirs(output_dir, exist_ok=True)

        # POSCAR
        from pymatgen.io.vasp.inputs import Poscar
        Poscar(structure).write_file(os.path.join(output_dir, "POSCAR"))

        # INCAR and KPOINTS
        vis.incar.write_file(os.path.join(output_dir, "INCAR"))
        vis.kpoints.write_file(os.path.join(output_dir, "KPOINTS"))

        return output_dir
    

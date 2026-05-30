import h5py
import numpy as np
import re
import argparse
import sys

# ================================================= #
# ATOM MAPPING LOGIC ---More can be added manually  #
# ================================================= #

class AtomMapper:
    """Handles the conversion of Amber/H5 names to chemical symbols."""
    AMBER_MAP = {
        "OW": "O", "HW": "H", "MW": "X", "CA": "C", "CT": "C",
        "CX": "C", "3C": "C", "2C": "C", "NZ": "N", "N3": "N",
        "O2": "O", "H1": "H", "H": "H", "CU": "Cu", "Na" : "Na",
        "ZN": "Zn" #"NA":"N", 
    }

    @staticmethod
    def get_element(name):
        name = name.strip().replace('\x00', '')
        if name in AtomMapper.AMBER_MAP:
            return AtomMapper.AMBER_MAP[name]
        
        # Fallback: regex to find first letters
        letters = "".join(re.findall(r'[a-zA-Z]+', name))
        return letters[0].upper() if letters else "H"

# ==========================================
# WRITER REGISTRY
# ==========================================

class Writers:
    """Contains various output formatters."""
    
    @staticmethod
    def to_xyz(filename, coords, atom_elements, unit_scale=10.0):
        n_frames, n_atoms, _ = coords.shape
        print(f"Writing {n_frames} frames to XYZ...")
        
        with open(filename, "w") as f:
            for frame_idx in range(n_frames):
                frame_coords = coords[frame_idx]
                
                # Safety check for empty frames (from your original code)
                if np.all(frame_coords == 0) or np.any(np.isnan(frame_coords)):
                    print(f"Reached end of valid data at frame {frame_idx}. Stopping.")
                    break

                scaled_coords = frame_coords * unit_scale
                f.write(f"{n_atoms}\n")
                f.write(f"Frame {frame_idx}\n")

                for i in range(n_atoms):
                    symbol = atom_elements[i]
                    x, y, z = scaled_coords[i]
                    f.write(f"{symbol:5} {x:12.6f} {y:12.6f} {z:12.6f}\n")
        print(f"Successfully saved: {filename}")

    @staticmethod
    def to_pdb(filename, coords, atom_elements, unit_scale=10.0):
        """Placeholder: Add your PDB logic here later."""
        print(f"PDB output requested for {filename}, but logic is not yet implemented.")

    @staticmethod
    def to_gro(filename, coords, atom_elements, unit_scale=10.0):
        """Placeholder: Add your GRO logic here later."""
        print(f"GRO output requested for {filename}, but logic is not yet implemented.")

# ==========================================
# MAIN CONVERTER CLASS
# ==========================================

class TrajectoryConverter:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.atom_elements = []
        self.coordinates = None

    def load_data(self):
        """Reads the specific H5 structure provided in your snippet."""
        with h5py.File(self.h5_path, "r") as f:
            # 1. Map Species to Elements
            raw_unique_names = f["parameters/vmd_structure/name"][:]
            unique_names = [n.decode('utf-8') for n in raw_unique_names]
            species_indices = f["particles/all/species"][:]
            
            self.atom_elements = [AtomMapper.get_element(unique_names[idx]) for idx in species_indices]

            # 2. Extract Coordinates
            # Note: We keep this as a dataset object or array
            self.coordinates = f["particles/all/position/value"][:] 

    def run_conversion(self, output_path, format_key):
        """Dispatches the data to the requested writer."""
        if not self.atom_elements:
            self.load_data()

        mapping = {
            "xyz": Writers.to_xyz,
            "pdb": Writers.to_pdb,
            "gro": Writers.to_gro
        }

        if format_key in mapping:
            mapping[format_key](output_path, self.coordinates, self.atom_elements)
        else:
            print(f"Error: Format '{format_key}' is not recognized.")

# ==========================================
# ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Modular H5 Trajectory Converter")

    # Required Input
    parser.add_argument("-i", "--input", type=str, required=True, 
                        help="Path to the input tra.h5 file")

    # Output Options
    parser.add_argument("-oxyz", metavar="FILE", type=str, help="Output to XYZ format")
    parser.add_argument("-opdb", metavar="FILE", type=str, help="Output to PDB format")
    parser.add_argument("-ogro", metavar="FILE", type=str, help="Output to GRO format")

    args = parser.parse_args()

    # Initialize converter
    conv = TrajectoryConverter(args.input)

    try:
        if args.oxyz:
            conv.run_conversion(args.oxyz, "xyz")
        if args.opdb:
            conv.run_conversion(args.opdb, "pdb")
        if args.ogro:
            conv.run_conversion(args.ogro, "gro")
            
        if not any([args.oxyz, args.opdb, args.ogro]):
            print("No output format specified. Use -oxyz <file> to generate output.")

    except FileNotFoundError:
        print(f"Error: The file '{args.input}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

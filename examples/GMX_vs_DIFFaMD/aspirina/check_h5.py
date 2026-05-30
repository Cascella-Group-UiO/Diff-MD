import h5py
import numpy as np

def inspect_full_h5(file_path):
    with h5py.File(file_path, "r") as f:
        print(f"{'='*60}")
        print(f"FULL H5 FILE DUMP: {file_path}")
        print(f"{'='*60}\n")

        # 1. Inspect Global Attributes (Box, n_molecules, etc.)
        print("--- GLOBAL ATTRIBUTES ---")
        for attr_name, attr_value in f.attrs.items():
            print(f"  {attr_name}: {attr_value}")
        print("\n")

        # 2. Function to print dataset details
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}Dataset: {name}")
                print(f"{indent}  Shape: {obj.shape} | Type: {obj.dtype}")

        print("--- FILE HIERARCHY ---")
        f.visititems(print_structure)
        print("\n")

        # 3. Print Data for Every Dataset
        print("--- DATA VALUES ---")
        for key in f.keys():
            data = f[key][:]
            print(f"\nDataset '{key}':")
            
            # Special handling for Strings (names, resnames)
            if data.dtype.kind in {'S', 'U'}:
                # Decode byte strings for readability
                decoded = [d.decode('utf-8') if isinstance(d, bytes) else d for d in data]
                print(np.array(decoded))
            
            # Special handling for coordinates/velocities (high-dimensional)
            elif len(data.shape) > 1:
                print(f" (Showing first frame of {data.shape[0]})")
                print(data[0]) # Print the first frame (all atoms)
            
            else:
                print(data)

        # 4. Physical Consistency Check
        if "charge" in f:
            q = f["charge"][:]
            print(f"\n{'='*60}")
            print(f"PHYSICS VALIDATION")
            print(f"{'='*60}")
            print(f"Total System Charge: {np.sum(q):.8f}")
            
            # Group charges by residue name to see where the "nonsense" is
            if "resnames" in f:
                res = [r.decode() for r in f["resnames"][:]]
                unique_res = np.unique(res)
                print("\nCharge sum by Residue Type:")
                for r_type in unique_res:
                    mask = np.array(res) == r_type
                    print(f"  {r_type:<5}: {np.sum(q[mask]):>10.4f}")

if __name__ == "__main__":
    # Ensure this matches the file name your converter produced
    inspect_full_h5("output.h5")

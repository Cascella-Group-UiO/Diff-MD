import h5py
import numpy as np

def inspect_full_h5(file_path):
    with h5py.File(file_path, "r") as f:
        print(f"{'='*60}")
        print(f"H5 INSPECTION: {file_path}")
        print(f"{'='*60}\n")

        # 1. THE BOX HUNTER (Looking for dimensions)
        print("--- BOX / UNIT CELL SEARCH ---")
        box_found = False
        
        # Search in Global Attributes
        box_attr_keywords = ['box', 'cell', 'unitcell', 'dimensions', 'pbc', 'lattice']
        for attr_name, attr_value in f.attrs.items():
            if any(k in attr_name.lower() for k in box_attr_keywords):
                print(f"  [FOUND IN ATTRS] {attr_name}: {attr_value}")
                box_found = True

        # Search in Dataset names
        for key in f.keys():
            if any(k in key.lower() for k in box_attr_keywords):
                print(f"  [FOUND AS DATASET] '{key}':")
                print(f"    Shape: {f[key].shape} | Values: {f[key][()]}")
                box_found = True
        
        if not box_found:
            print("  !! WARNING: No obvious 'box' or 'cell' info found in root !!")
        print("\n")

        # 2. PRINT ALL TOP-LEVEL FIELDS (Datasets & Groups)
        print("--- TOP-LEVEL FIELDS ---")
        fields = list(f.keys())
        print(f"Total fields: {len(fields)}")
        print(f"Field names: {fields}\n")

        # 3. FULL HIERARCHY
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                # We use .get() to avoid loading massive data into memory just for a print
                print(f"{indent}Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")

        print("--- DETAILED HIERARCHY ---")
        f.visititems(print_structure)
        print("\n")

        # 4. PREVIEW SAMPLES (Only first elements to keep it clean)
        print("--- DATA SAMPLES ---")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                data = f[key]
                print(f"\n{key}:")
                if data.shape == (): # Scalar
                    print(f"  Value: {data[()]}")
                elif len(data.shape) == 1: # 1D Array
                    print(f"  First 5 values: {data[:5]}")
                else: # High-dimensional (coords, etc.)
                    print(f"  First entry shape {data[0].shape}:")
                    print(data[0])

if __name__ == "__main__":
    # Change to your filename
    inspect_full_h5("output.h5")

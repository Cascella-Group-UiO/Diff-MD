import h5py
import numpy as np

file_path = "output.h5" # Change to your filename

with h5py.File(file_path, "r") as f:
    charges = f["charge"][:]
    names = f["names"][:]
    resnames = f["resnames"][:]
    
    print(f"--- H5 Inspection: {file_path} ---")
    print(f"Total Atoms: {len(charges)}")
    
    # 1. Check for Total Charge (Should be near 0.0 for a neutralized system)
    total_q = np.sum(charges)
    print(f"Net System Charge: {total_q:.6f}")
    
    # 2. Check for "Nonsense" (Are they all zeros? Are they massive numbers?)
    print(f"Charge Range: Min={np.min(charges):.4f}, Max={np.max(charges):.4f}")
    
    # 3. Print a sample to verify mapping
    print("\nSample Atom Data (First 10):")
    print(f"{'Index':>5} | {'Name':>5} | {'Res':>5} | {'Charge':>8}")
    print("-" * 40)
    for i in range(10):
        # We use .decode() because H5 strings are stored as bytes
        n = names[i].decode()
        r = resnames[i].decode()
        print(f"{i:>5} | {n:>5} | {r:>5} | {charges[i]:>8.4f}")

    # 4. Check specific common residues (e.g., Ions or Water)
    zn_indices = np.where(names == b'ZN')
    if len(zn_indices[0]) > 0:
        print(f"\nExample: Charge of first ZN atom: {charges[zn_indices[0][0]]}")

import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

def write_multicolumn_xvg(filename, time_data, properties_dict, title):
    """Writes multiple datasets sharing the same time axis into a multi-column XVG file."""
    with open(filename, 'w') as f:
        f.write(f'# Created by diff-aMD Energy Analyzer\n')
        f.write(f'@    title "{title}"\n')
        f.write(f'@    xaxis  label "Time (fs)"\n')
        f.write(f'@    yaxis  label "Energy / Value"\n')
        f.write(f'@TYPE xy\n')
        
        # Write the legend labels for each column
        prop_names = list(properties_dict.keys())
        for i, prop_name in enumerate(prop_names):
            f.write(f'@ s{i} legend "{prop_name}"\n')
            
        # Write the actual data columns: Time  Prop1  Prop2  Prop3...
        for i in range(len(time_data)):
            line = f"{time_data[i]:14.5f}"
            for p in prop_names:
                line += f" {properties_dict[p][i]:14.5f}"
            f.write(line + "\n")
            
    print(f"  [+] Successfully saved multi-column data to {filename}")

def diff_amd_energy(log_file):
    columns = [
        "step", "time_fs", "temp_K", "E_total_kJmol", "E_potential_kJmol", 
        "E_kin_kJmol", "E_LJ_kJmol", "E_elec_kJmol", "E_bond_kJmol", 
        "E_angle_kJmol", "E_torsional_kJmol", "E_improper_torsional_kJmol"
    ]

    try:
        print(f"Loading data from '{log_file}'...")
        data = np.loadtxt(log_file, comments='#')
    except Exception as e:
        print(f"\n[!] Error reading {log_file}: {e}")
        sys.exit(1)

    if len(data) == 0:
        print("\n[!] The log file is empty or formatted incorrectly.")
        sys.exit(1)

    num_steps = len(data)
    time_data = data[:, 1] # Time is always column 1
    simulated_time = time_data[-1] - time_data[0] 
    analyzable_props = columns[2:]

    # Calculate the values first for cleaner code
    time_ps = simulated_time / 1000
    time_ns = simulated_time / 1_000_000

    while True:
        print("\n" + "="*55)
        print(" diff-aMD Energy Analysis ".center(55, "="))
        print("="*55)
        for i, prop in enumerate(analyzable_props, start=1):
            print(f"  {i:>2}. {prop}")
        print("-" * 55)
        print("   0. Exit (or type 'q')")
        
        # Ask for a comma-separated list, allow 'q' to exit
        choice_str = input("\nEnter property numbers separated by commas (e.g., 2,3,4) or 0/q to exit: ").strip().lower()
        
        if choice_str in ['0', 'q']:
            print("Exiting diff-aMD energy analyzer. Ciao Ciao!")
            break
            
        try:
            # Parse the comma-separated input into a list of integers
            choices = [int(c.strip()) for c in choice_str.split(',')]
            
            selected_properties = {}
            print("\n" + "*"*60)
            print(f" RESULTS OVER {num_steps} STEPS ({simulated_time:.2f} fs | {time_ps:.4f} ps | {time_ns:.6f} ns)")
            print("*"*60)
            
            valid_choices = True
            for choice_idx in choices:
                if 1 <= choice_idx <= len(analyzable_props):
                    prop_name = analyzable_props[choice_idx - 1]
                    col_idx = choice_idx + 1
                    y_data = data[:, col_idx]
                    
                    selected_properties[prop_name] = y_data
                    
                    # Print stats for each selected property
                    print(f" {prop_name:>25}: Mean = {np.mean(y_data):>10.4f} ± {np.std(y_data):<10.4f}")

                    # --- Modular Drift calculation for Total and Potential Energy ---
                    if prop_name in ["E_total_kJmol", "E_potential_kJmol"] and len(time_data) > 1:
                        # polyfit returns [slope, intercept]. We only want the slope (kJ/mol/fs).
                        slope_fs, _ = np.polyfit(time_data, y_data, 1)
                        
                        # Dynamically scale the drift based on total simulation time
                        if simulated_time >= 1_000_000:
                            drift_val = slope_fs * 1_000_000
                            drift_unit = "ns"
                        elif simulated_time >= 1_000:
                            drift_val = slope_fs * 1_000
                            drift_unit = "ps"
                        else:
                            drift_val = slope_fs
                            drift_unit = "fs"
                            
                        print(f" {'-> Drift (Slope)':>25}: {drift_val:>10.4f} kJ/mol/{drift_unit}")
                        
                else:
                    print(f"[!] Invalid choice '{choice_idx}'. Skipping.")
                    valid_choices = False
            
            print("*"*60)

            if not selected_properties:
                continue # None of the choices were valid

            # --- Plotting & Export Sub-Menu ---
            action = input(f"\nOptions for selected properties: [P]lot, [X]VG export, [B]oth, or [Enter] to skip: ").strip().lower()
            
            # Create a combined name for the file, e.g., "Etotal_Epotential.xvg"
            combo_name = "_".join([p.split('_')[1] for p in selected_properties.keys()])
            
            if action in ['x', 'b']:
                out_name = f"multi_{combo_name}.xvg"
                write_multicolumn_xvg(out_name, time_data, selected_properties, f"Energy Analysis: {combo_name}")

            if action in ['p', 'b']:
                print(f"  [+] Opening combined plot...")
                plt.figure(figsize=(10, 6))
                
                for prop_name, y_data in selected_properties.items():
                    plt.plot(time_data, y_data, label=prop_name, linewidth=1.2, alpha=0.9)
                    
                plt.xlabel('Time (fs)', fontweight='bold')
                plt.ylabel('Energy / Value', fontweight='bold')
                plt.title(f'Multi-Property Comparison vs Time')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plt.show()

        except ValueError:
            print("\n[!] Invalid input format. Please enter numbers separated by commas (e.g., 2, 4, 5).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot multiple energies from a .log file.")
    parser.add_argument("-f", "--file", required=True, help="Input log file (.log)")
    args = parser.parse_args()
    diff_amd_energy(args.file)

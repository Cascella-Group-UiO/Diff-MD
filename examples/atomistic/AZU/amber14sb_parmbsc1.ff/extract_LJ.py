import sys

def extract_plain_lj(filename):
    lj_params = []
    in_atomtypes = False

    try:
        with open(filename, 'r') as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line or clean_line.startswith(';'):
                    continue

                # Detect [ atomtypes ] section
                if clean_line.startswith('['):
                    section_name = clean_line.strip('[] ').lower()
                    in_atomtypes = (section_name == 'atomtypes')
                    continue

                # Exit if a new section starts
                if in_atomtypes and clean_line.startswith('['):
                    break

                if in_atomtypes:
                    parts = clean_line.split(';')[0].split()
                    
                    # Target name (0), sigma (5), and epsilon (6)
                    if len(parts) >= 7:
                        lj_params.append([parts[0], parts[5], parts[6]])

        # Print with exact spacing format
        print("LJ_type_param = [")
        for p in lj_params:
            name = f"'{p[0]}',"
            # :.2e ensures the scientific notation matches your provided style
            sigma = f"{float(p[1]):.2e},"
            eps = f"{float(p[2]):.2e}"
            
            # Formatting to match the provided alignment:
            # [ 'NAME',  SIGMA, EPSILON ]
            print(f"    [ {name:<8} {sigma:<10} {eps:<10} ],")
        print("]")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")

if __name__ == "__main__":
    # Ensure this matches your itp filename
    extract_plain_lj('ffnonbonded.itp')

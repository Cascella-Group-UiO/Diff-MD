import numpy as np


class GroAtom:
    def __init__(
        self, resid, resname, atom_name, index, x, y, z, vx=0.0, vy=0.0, vz=0.0
    ):
        self.resid = resid
        self.resname = resname
        self.atom_name = atom_name
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz

    @classmethod
    def parse_line(cls, line):
        line = line.rstrip("\n")
        line_length = len(line)
        if line_length not in (44, 68):
            raise ValueError(
                f"Gro file line not formatted correctly:\n"
                f"{line}"
                f"\nThe line length is {line_length}, "
                "while it should be 44 for a GRO file containing positions only "
                "or 68 for a GRO file containing both positions and velocities."
            )
        resid = int(line[:5])
        resname = line[5:10].strip()
        atom_name = line[10:15].strip()
        index = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        vx = float(line[44:52]) if line_length == 68 else 0.0
        vy = float(line[52:60]) if line_length == 68 else 0.0
        vz = float(line[60:68]) if line_length == 68 else 0.0
        return cls(resid, resname, atom_name, index, x, y, z, vx, vy, vz)


def load_gro(filename: str) -> tuple[list[GroAtom], np.ndarray]:
    """Parse gro file"""
    with open(filename, "r") as infile:
        lines = infile.readlines()

    if len(lines) < 3:
        raise ValueError(f"GRO file '{filename}' is incomplete: expected title, atom count, and box line.")

    try:
        n_atoms = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"GRO file '{filename}' has an invalid atom-count line: {lines[1].strip()!r}.") from exc

    if len(lines) < n_atoms + 3:
        raise ValueError(
            f"GRO file '{filename}' is incomplete: declares {n_atoms} atoms but has "
            f"only {max(0, len(lines) - 3)} atom lines."
        )

    box_tokens = lines[n_atoms + 2].split()
    if len(box_tokens) != 3:
        raise ValueError(
            f"GRO file '{filename}' has an unsupported box line with {len(box_tokens)} values; "
            "expected exactly 3 orthorhombic box lengths."
        )

    atom_list = []
    box_size = np.array(box_tokens, dtype=float)
    for line in lines[2:n_atoms + 2]:
        atom_list.append(GroAtom.parse_line(line))
    print(f"GRO file {filename} loaded... ")
    return atom_list, box_size

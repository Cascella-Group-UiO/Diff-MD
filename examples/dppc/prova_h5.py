import h5py

def get_atom_coords(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['particles/all/position/value'][:]
        # Create a dictionary: {atom_index: [x, y, z] for frame 0}
        first_frame_dict = {i: pos for i, pos in enumerate(data[0])}
        return data, first_frame_dict

traj_array, atom_map = get_atom_coords('simulation.h5')
print(f"Atom 5 coordinates at Frame 0: {atom_map[5]}")

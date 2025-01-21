import numpy as np
from jax import jit
from jax import numpy as jnp

def verlet_list(positions, box, rv, n_particles):
    """
    Create a neighbor list for particles in a 3D system based on their positions.

    Parameters:
        positions (array): Array of particle positions (n_particle x 3).
        box (array): Full size of the simulation box in each dimension (3,).
        rv (float): Cutoff distance for neighbor determination.
        n_particle (int): Number of particles.

    Returns:
        list_neighbors (dict): Neighbor list for each particle.
    """

    # Initialize neighbor list
    list_neighbors = {i: [] for i in range(n_particles)} # Dictionary for neighbor pairs

    # Iterate over all particle pairs
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # Compute the position difference vector
            r_vec = positions[i] - positions[j]

            # Apply periodic boundary conditions (nearest image convention)
            r_vec = r_vec - box * np.round(r_vec / box)

            # Compute the squared distance between the particles
            r_squared = np.dot(r_vec, r_vec)

            # Check if the distance is within the cutoff radius
            if r_squared < rv ** 2:
                # Update neighbor list
                list_neighbors[i].append(j)
                list_neighbors[j].append(i)

    return list_neighbors


@jit
def apply_cutoff(r, r_norm, sigma, epsilon, rc):
    """
    Applies cutoff to interactions based on distance.

    Args:
        r: Array of particle pair distances.
        r_norm: Array of norms of particle pair distances.
        sigma: Array of sigma values for particle pairs.
        epsilon: Array of epsilon values for particle pairs.
        rc: Cutoff distance.

    Returns:
        Tuple containing filtered r, sigma, and epsilon arrays.
    """
    # Use jnp.less to create a boolean mask 
    mask = jnp.less(r_norm, rc) 
    
    # Use jnp.where with the mask to get indices
    cut_indx = jnp.where(mask, size=r_norm.shape[0], fill_value=-1)[0]
    
    # Apply the mask to r, sigma and epsilon using jnp.take
    r_values = jnp.take(r, cut_indx, axis=0)
    sigma_values = jnp.take(sigma, cut_indx)
    epsilon_values = jnp.take(epsilon, cut_indx)
    epsilon_values = jnp.where(cut_indx == -1, 0, epsilon_values)

    return r_values, sigma_values, epsilon_values
from jax import jit, Array
from jax import numpy as jnp
import numpy as np
from typing import Tuple


@jit
def nlist(positions, box_size, r_cut, i, j):
  
  """
  Brute force neighbor list
  At least 10 times slower than Vesin
  """

  positions_i = jnp.expand_dims(positions, axis=1)

  r_vec = positions_i - positions
  r_vec = r_vec - box_size * jnp.around(r_vec / box_size)
  r = jnp.linalg.norm(r_vec, axis=2)

  # mask = jnp.less(r, r_cut)
  mask = jnp.triu(r < r_cut, k=1)

  i_idx, j_idx = jnp.where(mask, size=i.shape[0], fill_value=-1)

  i = i.at[...].set(i_idx)
  j = j.at[...].set(j_idx)

  return i, j


@jit
def apply_cutoff(
    r: Array,
    r_norm: Array,
    sigma: Array,
    epsilon: Array, 
    neigh_i: Array,
    neigh_j: Array,
    rc: float
) -> Tuple[float, Array]:

    # Use jnp.less to create a boolean mask
    mask = jnp.less(r_norm, rc)
    # Account for padding in nlists
    mask = jnp.where(neigh_i==-1, False, mask)

    # Use jnp.where with the mask to get indices
    cut_indx = jnp.where(mask, size=r_norm.shape[0], fill_value=-1)[0]

    # Apply the mask using jnp.take
    r_values = jnp.take(r, cut_indx, axis=0)
    # r_norm = jnp.take(r_norm, cut_indx)
    # print(sigma.shape)
    sigma_values = jnp.take(sigma, cut_indx)
    epsilon_values = jnp.take(epsilon, cut_indx)
    # Set epsilon to 0 for pair interactions outside the cutoff
    epsilon_values = jnp.where(cut_indx == -1, 0, epsilon_values)

    # Update neighbor list
    neigh_i_mod = jnp.take(neigh_i, cut_indx)
    neigh_j_mod = jnp.take(neigh_j, cut_indx)
    

    return r_values, sigma_values, epsilon_values, neigh_i_mod, neigh_j_mod


@jit
def apply_cutoff_elec(
    r: Array,
    r_norm: Array,
    q_i: Array,
    q_j: Array,
    neigh_i: Array,
    neigh_j: Array,
    rc: float
) -> Tuple[float, Array]:

    # Use jnp.less to create a boolean mask
    mask = jnp.less(r_norm, rc)
    # Account for padding in nlists
    mask = jnp.where(neigh_i==-1, False, mask)

    # Use jnp.where with the mask to get indices
    cut_indx = jnp.where(mask, size=r_norm.shape[0], fill_value=-1)[0]

    # Apply the mask using jnp.take
    r_values = jnp.take(r, cut_indx, axis=0)
    r_norm = jnp.take(r_norm, cut_indx)
    q_i = jnp.take(q_i, cut_indx)
    q_j = jnp.take(q_j, cut_indx)

    # Set q to 0 for pair interactions outside the cutoff
    q_i = jnp.where(cut_indx == -1, 0, q_i)
    q_j = jnp.where(cut_indx == -1, 0, q_j)

    # Update neighbor list
    neigh_i_mod = jnp.take(neigh_i, cut_indx)
    neigh_j_mod = jnp.take(neigh_j, cut_indx)

    return r_values, r_norm, q_i, q_j, neigh_i_mod, neigh_j_mod

@jit
def apply_nlist(
    neigh_i: Array,
    neigh_j: Array,
    positions: Array,
    box_size: Array,
    sigma: Array,
    epsilon: Array,
    types: Array    
):
    # For calculation of pair potential energy
    i = jnp.take(positions, neigh_i, axis=0)
    j = jnp.take(positions, neigh_j, axis=0)
    r_vec = i - j
    r_vec = r_vec - box_size * jnp.around(r_vec / box_size)
    r = jnp.linalg.norm(r_vec, axis=1)
    q_i = None
    q_j = None

    s_ij = sigma[types[neigh_i[:]], types[neigh_j[:]]]
    e_ij = epsilon[types[neigh_i[:]], types[neigh_j[:]]]
    # print(s_ij.shape)

    return r_vec, r, neigh_i, neigh_j, q_i, q_j, s_ij, e_ij

@jit
def apply_nlist_elec(
    neigh_i: Array,
    neigh_j: Array,
    positions: Array,
    charges: Array,
    box_size: Array,
    sigma: Array,
    epsilon: Array,
    types: Array    
):
    # For calculation of pair potential energy
    i = jnp.take(positions, neigh_i, axis=0)
    j = jnp.take(positions, neigh_j, axis=0)
    r_vec = i - j
    r_vec = r_vec - box_size * jnp.around(r_vec / box_size)
    r = jnp.linalg.norm(r_vec, axis=1)
    q_i = charges[neigh_i]
    q_j = charges[neigh_j]

    s_ij = sigma[types[neigh_i[:]], types[neigh_j[:]]]
    e_ij = epsilon[types[neigh_i[:]], types[neigh_j[:]]]

    return r_vec, r, neigh_i, neigh_j, q_i, q_j, s_ij, e_ij

# NOTE: Try to jit this function
# @jit
def exclude_bonded_neighbors(neigh_i, neigh_j, bonded_i, bonded_j):
    bonded_pairs = jnp.vstack((jnp.column_stack((bonded_i, bonded_j)),
                               jnp.column_stack((bonded_j, bonded_i))))
    neighbor_pairs = jnp.column_stack((neigh_i, neigh_j))

    # Create a mask that keeps only non-bonded neighbors
    # Compare each neighbor pair to all bonded pairs
    mask = ~jnp.any(jnp.all(neighbor_pairs[:, None, :] == bonded_pairs, axis=-1), axis=1)

    filtered_neigh_i = neigh_i[mask]
    filtered_neigh_j = neigh_j[mask]

    return filtered_neigh_i, filtered_neigh_j
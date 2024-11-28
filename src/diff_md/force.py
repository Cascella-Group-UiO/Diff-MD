import jax.numpy as jnp
from jax import debug, grad, jacrev, jit, value_and_grad, vmap


@jit
def harmonic_potential(x, x0, k):
    return 0.5 * k * (x - x0) ** 2


# Bonds
@jit
def bond_potential(ra, rb, r0, k, box):
    rab = ra - rb
    rab -= box * jnp.around(rab / box)
    rabnorm = jnp.linalg.norm(rab)
    return harmonic_potential(rabnorm, r0, k), rab


@jit
def get_bond_energy_and_forces(forces, pos, box, atom1, atom2, r0, k):
    ra = pos[atom1]
    rb = pos[atom2]
    vbond_grad = vmap(value_and_grad(bond_potential, has_aux=True), (0, 0, 0, 0, None))
    (energies, rijs), grads = vbond_grad(ra, rb, r0, k, box)

    bond_pressure = jnp.sum(-grads * rijs, axis=0)
    forces = forces.at[...].set(0.0)
    forces = forces.at[atom1].add(-grads)
    forces = forces.at[atom2].add(grads)
    return jnp.sum(energies), forces, bond_pressure


# Angles
def get_angle(ra, rb, rc, box):
    ab = ra - rb
    cb = rc - rb

    ab -= box * jnp.around(ab / box)
    cb -= box * jnp.around(cb / box)

    u_ab = ab / jnp.linalg.norm(ab)
    u_cb = cb / jnp.linalg.norm(cb)

    cos_angle = jnp.dot(u_ab, u_cb)
    condition = jnp.isclose(cos_angle * cos_angle, 1.0)
    safe_cos = jnp.where(condition, jnp.rint(cos_angle), cos_angle)
    # return jnp.where(condition, jnp., jnp.arccos(safe_cos)), condition, (ab, cb)
    return jnp.arccos(safe_cos), condition, (ab, cb)


@jit
def angle_potential(ra, rb, rc, theta_0, k, box):
    theta, condition, bond_vectors = get_angle(ra, rb, rc, box)
    return (
        jnp.where(condition, 0.0, harmonic_potential(theta, theta_0, k)),
        (bond_vectors, theta),
    )


@jit
def get_angle_energy_and_forces(forces, pos, box, atom1, atom2, atom3, theta_0, k):
    ra = pos[atom1]
    rb = pos[atom2]
    rc = pos[atom3]

    vangle_grad = vmap(
        value_and_grad(angle_potential, (0, 2), has_aux=True), (0, 0, 0, 0, 0, None)
    )
    (energies, ((rijs, rkjs), _)), (grad_ra, grad_rc) = vangle_grad(
        ra, rb, rc, theta_0, k, box
    )

    angle_pressure = jnp.sum(-grad_ra * rijs - grad_rc * rkjs, axis=0)
    forces = forces.at[...].set(0)
    forces = forces.at[atom1].add(-grad_ra)
    forces = forces.at[atom2].add(grad_ra + grad_rc)
    forces = forces.at[atom3].add(-grad_rc)
    return jnp.sum(energies), forces, angle_pressure


# Dihedrals
def get_dihedral_angle(ra, rb, rc, rd, box):
    f = ra - rb
    g = rb - rc
    h = rd - rc
    k = f + g  # needed for the virial

    f -= box * jnp.around(f / box)  # r_ab
    g -= box * jnp.around(g / box)  # r_bc
    h -= box * jnp.around(h / box)  # r_dc
    k -= box * jnp.around(k / box)  # r_ac

    v = jnp.cross(f, g)
    w = jnp.cross(h, g)
    gn = jnp.linalg.norm(g)

    cos_phi = jnp.dot(v, w)
    sin_phi = jnp.dot(w, f) * gn

    # Safe grad calculation: set angle to 0 when the beads are collinear
    condition = jnp.isclose(cos_phi, 0.0) == jnp.isclose(sin_phi, 0.0)
    safe_cos = jnp.where(condition, 1.0, cos_phi)
    return jnp.where(condition, 0.0, jnp.arctan2(sin_phi, safe_cos)), (k, g, h)


@jit
def cbt_potential(ra, rb, rc, rd, coeff, last, box):
    phi, bond_vectors = get_dihedral_angle(ra, rb, rc, rd, box)
    series_len = jnp.arange(5.0)

    def cosine_series_element(coeff_n, phase_n, phi, n):
        return coeff_n * (1 + jnp.cos(n * phi - phase_n))

    cosine_series = vmap(cosine_series_element, (0, 0, None, 0))

    # V_prop coefficients
    energy_dih = jnp.sum(cosine_series(coeff[0], coeff[1], phi, series_len))

    # Angle force constant
    k_phi = jnp.sum(cosine_series(coeff[2], coeff[3], phi, series_len))

    # Reference angle
    check_empty = jnp.any(coeff[4:])  # False if all are zeros
    gamma_0 = jnp.where(
        check_empty,
        jnp.sum(cosine_series(coeff[4], coeff[5], phi, series_len)),
        1.85 - 0.227 * jnp.cos(phi - 0.785),
    )

    energy_ang, (_, theta) = angle_potential(ra, rb, rc, gamma_0, k_phi, box)

    # NOTE: not sure how to convert this to save the angle without writing another function
    last_angle_energy = jnp.where(
        # fmt: off
        last == 1,
        angle_potential(rb, rc, rd, gamma_0, k_phi, box)[0], # only get energy, discard vectors and angle
        0.0,
    )
    return energy_dih + energy_ang + last_angle_energy, ((phi, theta), bond_vectors)


@jit
def get_dihedral_energy_and_forces(
    forces, pos, box, atom1, atom2, atom3, atom4, coeff, last
):
    ra = pos[atom1]
    rb = pos[atom2]
    rc = pos[atom3]
    rd = pos[atom4]

    dih_grad = vmap(
        value_and_grad(cbt_potential, (0, 1, 2, 3), has_aux=True),
        (0, 0, 0, 0, 0, 0, None),
    )
    (energies, angles, (r_ac, r_bc, r_dc)), grads = dih_grad(
        ra, rb, rc, rd, coeff, last, box
    )

    dihedral_pressure = jnp.sum(-grads[0] * r_ac - grads[1] * r_bc - grads[3] * r_dc)
    forces = forces.at[...].set(0)
    forces = forces.at[atom1].add(-grads[0])
    forces = forces.at[atom2].add(-grads[1])
    forces = forces.at[atom3].add(-grads[2])
    forces = forces.at[atom4].add(-grads[3])
    return jnp.sum(energies), forces, angles, dihedral_pressure


# Improper dihedrals
def improper_potential(ra, rb, rc, rd, phi_0, k, box):
    phi = get_dihedral_angle(ra, rb, rc, rd, box)
    energy = harmonic_potential(phi, phi_0, k)
    return energy


@jit
def get_impropers_energy_and_forces(
    forces, pos, box, atom1, atom2, atom3, atom4, coeff, last
):
    ra = pos[atom1]
    rb = pos[atom2]
    rc = pos[atom3]
    rd = pos[atom4]

    improper_grad = vmap(
        value_and_grad(improper_potential, (0, 1, 2, 3)),
        (0, 0, 0, 0, 0, 0, None),
    )
    energies, grads = improper_grad(ra, rb, rc, rd, coeff, last, box)

    forces = forces.at[...].set(0)
    forces = forces.at[atom1].add(-grads[0])
    forces = forces.at[atom2].add(-grads[1])
    forces = forces.at[atom3].add(-grads[2])
    forces = forces.at[atom4].add(-grads[3])
    return jnp.sum(energies), forces


# https://doi.org/10.1021/ct400219n
# @jit
# def triplet_angle(ra, rb, rc, box):
#     ab = ra - rb
#     cb = rc - rb

#     ab -= box * jnp.around(ab / box)
#     cb -= box * jnp.around(cb / box)

#     u_ab = ab / jnp.linalg.norm(ab)
#     u_cb = cb / jnp.linalg.norm(cb)

#     cos_theta = jnp.dot(u_ab, u_cb)

#     # condition = cos_theta * cos_theta < 1.0
#     # safe_cos = jnp.where(condition, cos_theta, 0.0)
#     return jnp.arccos(cos_theta)


# def cos_pot_series(phi, a, n):
#     return a * jnp.cos(phi) ** n


# def cbt_potential(ra, rb, rc, rd, k, coeff, box):
#     g = rb - rc
#     h = rd - rc

#     f -= box * jnp.around(f / box)
#     g -= box * jnp.around(g / box)
#     h -= box * jnp.around(h / box)

#     v = jnp.cross(f, g)
#     w = jnp.cross(h, g)
#     gn = jnp.linalg.norm(g)

#     cosphi = jnp.dot(v, w)
#     sinphi = jnp.dot(w, f) * gn

#     cond = jnp.sum(jnp.logical_not(jnp.array([cosphi, sinphi]))) == 2
#     safe_cos = jnp.where(cond, 1.0, cosphi)
#     phi = jnp.where(cond, 0.0, jnp.arctan2(sinphi, safe_cos))
#
#     theta_0 = get_triplet_angle(ra, rb, rc, box)
#     theta_1 = get_triplet_angle(rb, rc, rd, box)
#     series = vmap(cos_pot_series, (None, 0, 0))
#     cos_term = jnp.sum(series(phi, coeff, jnp.arange(4)))
#     return k * jnp.sin(theta_0) ** 3 * jnp.sin(theta_1) ** 3 * cos_term


# Dipole reconstruction
# @jit
def theta_ang(gamma):
    """θ(γ) functional form"""
    return -1.607 * gamma + 0.094 + 1.883 / (1.0 + jnp.exp((gamma - 1.73) / 0.025))


@jit
def pw_theta(gamma):
    return jnp.piecewise(
        gamma,
        [gamma <= jnp.radians(90), gamma >= jnp.radians(108)],
        [lambda x: 1.977 - 1.607 * x, lambda x: 0.095 - 1.607 * x, theta_ang],
    )


@jit
def dipole_reconstruction(ra, rb, rc, box):
    """Returns the half dipole direction vector and dipole charge positions"""
    phi = 1.392947  # Not to be confused with the dihedral angle
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    gamma, _, (rab, rcb) = get_angle(ra, rb, rc, box)
    theta = pw_theta(gamma)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    u_ab = rab / jnp.linalg.norm(rab)
    u_cb = rcb / jnp.linalg.norm(rcb)
    n_vec = jnp.cross(u_ab, u_cb) / jnp.sin(gamma)
    m_vec = jnp.cross(n_vec, u_cb)

    # Direction vector
    d_dip = cos_phi * rcb + sin_phi * (cos_theta * n_vec + sin_theta * m_vec)

    # Dipole charge positions
    delta = 0.3  # charge distance
    r_zero = rb + 0.5 * rcb
    dipole = 0.5 * delta * d_dip
    dipole_plus = r_zero + dipole
    dipole_minus = r_zero - dipole
    return d_dip, (dipole_plus, dipole_minus)


@jit
def get_protein_dipoles(pos, box, atom1, atom2, atom3):
    ra = pos[atom1]
    rb = pos[atom2]
    rc = pos[atom3]

    get_matrices = vmap(
        jacrev(dipole_reconstruction, (0, 1, 2), has_aux=True), (0, 0, 0, None)
    )

    transfer_matrices, dipole_positions = get_matrices(ra, rb, rc, box)
    return transfer_matrices, jnp.mod(jnp.vstack(dipole_positions), box)


@jit
def dipole_force_transfer(sum_force, diff_force, matrix_a, matrix_b, matrix_c):
    # dipole_distance = 0.3
    force_a = 0.3 * matrix_a @ diff_force
    force_b = 0.3 * matrix_b @ diff_force + 0.5 * sum_force
    force_c = 0.3 * matrix_c @ diff_force + 0.5 * sum_force
    return force_a, force_b, force_c


@jit
def redistribute_dipole_forces(forces, dip_forces, trans_matrices, atom1, atom2, atom3):
    """Redistribute electrostatic forces calculated from ghost dipole point charges
    to the backcone atoms of the protein."""
    vmap_transfer = vmap(dipole_force_transfer, (0, 0, 0, 0, 0))
    fa, fb, fc = vmap_transfer(dip_forces[0], dip_forces[1], *trans_matrices)

    forces = forces.at[...].set(0)
    forces = forces.at[atom1].add(fa)
    forces = forces.at[atom2].add(fb)
    forces = forces.at[atom3].add(fc)
    return forces

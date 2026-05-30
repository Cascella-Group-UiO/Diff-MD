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


def get_bond_energy_and_forces(forces, pos, box, atom1, atom2, r0, k,
                               compute_pressure=True):
    ra = pos[atom1]
    rb = pos[atom2]
    vbond_grad = vmap(value_and_grad(bond_potential, has_aux=True), (0, 0, 0, 0, None))
    (energies, rijs), grads = vbond_grad(ra, rb, r0, k, box)

    if compute_pressure:
        bond_pressure = jnp.sum(-grads * rijs, axis=0)
    else:
        bond_pressure = jnp.zeros(3)
    forces = forces.at[...].set(0.0)
    forces = forces.at[atom1].add(-grads)
    forces = forces.at[atom2].add(grads)
    return jnp.sum(energies), forces, bond_pressure


# ── Angles ─────────────────────────────────────────────────────────────
# GROMACS angle function types implemented:
#   1  harmonic in theta    V = (1/2) k (theta - theta_0)^2
#   2  G96 cosine-based     V = (1/2) k (cos theta - cos theta_0)^2
#  10  restricted bending   V = (1/2) k (cos theta - cos theta_0)^2 / sin^2(theta)
#
# Two scalar potentials live below.  The wrapper
# ``get_angle_energy_and_forces`` switches between them at Python time on
# the static ``only_harmonic`` flag (set from
# ``Topology.angle_uses_only_harmonic`` upstream):
#
#   * ``_angle_potential_harmonic`` — single arithmetic chain, used for
#     atomistic FFs (amber_like, amber19sb) and any Martini system whose
#     angle table contains only type 1 rows.  XLA compiles a graph
#     bit-identical to the pre-2026-05-13 legacy.
#   * ``_angle_potential_multitype`` — per-row ``jnp.where`` dispatch over
#     all three types, used when at least one row is G96 or ReB.  Costs
#     ~3× the per-angle compute of the harmonic-only path, but bonded
#     angle work is well under 1% of step cost so the absolute hit is
#     negligible.
#
# Numerical care (both paths):
#   * cos(theta) is clipped to (-1+eps, 1-eps) before arccos so the
#     type-1 branch keeps a finite gradient at near-linear configurations
#     (where arccos's derivative diverges).
#   * sin^2(theta) is clamped to a per-precision floor (1e-7 in f64,
#     1e-4 in f32) before dividing in the ReB branch.  Mirrors the
#     GROMACS source numerical safeguard for theta -> pi.  The clamp's
#     subgradient is zero in the clamped region — the ReB force vanishes
#     instead of diverging there, which is fine for stable MD but a
#     deliberate deviation from the GROMACS analytic limit.

_REB_SIN2_EPS_F64 = 1e-7
_REB_SIN2_EPS_F32 = 1e-4


def get_angle(ra, rb, rc, box):
    """Theta + bond vectors with PBC.

    `theta` is computed via clipped arccos so the derivative stays
    finite at near-linear configurations.  The three-value return shape
    keeps older callers such as dipole reconstruction compatible.
    """
    ab = ra - rb
    cb = rc - rb

    ab -= box * jnp.around(ab / box)
    cb -= box * jnp.around(cb / box)

    u_ab = ab / jnp.linalg.norm(ab)
    u_cb = cb / jnp.linalg.norm(cb)

    cos_theta = jnp.dot(u_ab, u_cb)
    condition = jnp.isclose(cos_theta * cos_theta, 1.0)
    eps_clip = jnp.finfo(cos_theta.dtype).eps
    safe_cos = jnp.clip(cos_theta, -1.0 + eps_clip, 1.0 - eps_clip)
    return jnp.arccos(safe_cos), condition, (ab, cb)


@jit
def _angle_potential_harmonic(ra, rb, rc, theta_0, k, box):
    """Harmonic-in-theta angle potential — the fast path.

    Returns (energy, ((ab, cb), theta)).  Same aux shape as the
    multi-type kernel so the CBT dihedral and any future consumer can
    rely on a stable contract.
    """
    ab = ra - rb
    cb = rc - rb
    ab -= box * jnp.around(ab / box)
    cb -= box * jnp.around(cb / box)

    u_ab = ab / jnp.linalg.norm(ab)
    u_cb = cb / jnp.linalg.norm(cb)
    cos_theta = jnp.dot(u_ab, u_cb)

    eps_clip = jnp.finfo(cos_theta.dtype).eps
    safe_cos = jnp.clip(cos_theta, -1.0 + eps_clip, 1.0 - eps_clip)
    theta = jnp.arccos(safe_cos)
    energy = 0.5 * k * (theta - theta_0) ** 2
    return energy, ((ab, cb), theta)


@jit
def _angle_potential_multitype(ra, rb, rc, theta_0, k, angle_type, box):
    """Per-row type-dispatched angle potential.

    Computes all three GROMACS energy expressions and selects the live
    one with ``jnp.where``.  Reverse-mode AD traces through every branch
    even when only one is live — that's the price of a uniform-shape
    vmap with per-row dispatch.
    """
    ab = ra - rb
    cb = rc - rb
    ab -= box * jnp.around(ab / box)
    cb -= box * jnp.around(cb / box)

    u_ab = ab / jnp.linalg.norm(ab)
    u_cb = cb / jnp.linalg.norm(cb)
    cos_theta = jnp.dot(u_ab, u_cb)

    sin2_theta = 1.0 - cos_theta * cos_theta

    eps_clip = jnp.finfo(cos_theta.dtype).eps
    safe_cos = jnp.clip(cos_theta, -1.0 + eps_clip, 1.0 - eps_clip)
    theta = jnp.arccos(safe_cos)
    e_harm = 0.5 * k * (theta - theta_0) ** 2

    cos_theta_0 = jnp.cos(theta_0)
    dcos = cos_theta - cos_theta_0
    e_g96 = 0.5 * k * dcos * dcos

    eps_sin2 = (_REB_SIN2_EPS_F64 if cos_theta.dtype == jnp.float64
                else _REB_SIN2_EPS_F32)
    sin2_safe = jnp.maximum(sin2_theta, eps_sin2)
    e_reb = 0.5 * k * dcos * dcos / sin2_safe

    energy = jnp.where(
        angle_type == 1,
        e_harm,
        jnp.where(angle_type == 2, e_g96, e_reb),
    )
    return energy, ((ab, cb), theta)


# Back-compat alias — CBT dihedral and any external caller can still
# import ``angle_potential`` and get the multi-type kernel.  New code
# should call ``_angle_potential_harmonic`` directly when it knows the
# row is type 1.
angle_potential = _angle_potential_multitype


def get_angle_energy_and_forces(forces, pos, box, atom1, atom2, atom3,
                                theta_0, k, angle_type=None,
                                only_harmonic=False,
                                compute_pressure=True):
    """Bonded-angle energy, atomic forces, and virial pressure.

    Parameters
    ----------
    only_harmonic
        Python-time switch.  When ``True`` (atomistic FFs and any
        Martini system whose angle table is pure type 1, as detected by
        ``Topology.angle_uses_only_harmonic`` upstream), the kernel
        compiles to a single harmonic chain — bit-identical to the
        pre-2026-05-13 legacy graph.  When ``False``, the multi-type
        dispatch kernel runs and ``angle_type`` (per-row int array of 1,
        2, or 10) selects the energy expression on each row.
    angle_type
        Per-row GROMACS function type.  Ignored when
        ``only_harmonic=True``.  Defaults to all-1 when omitted, which
        keeps pre-multi-type test fixtures bit-identical.
    """
    ra = pos[atom1]
    rb = pos[atom2]
    rc = pos[atom3]

    if only_harmonic:
        vangle_grad = vmap(
            value_and_grad(_angle_potential_harmonic, (0, 2), has_aux=True),
            (0, 0, 0, 0, 0, None),
        )
        (energies, ((rijs, rkjs), _)), (grad_ra, grad_rc) = vangle_grad(
            ra, rb, rc, theta_0, k, box,
        )
    else:
        if angle_type is None:
            angle_type = jnp.ones_like(atom1, dtype=jnp.int32)
        vangle_grad = vmap(
            value_and_grad(_angle_potential_multitype, (0, 2), has_aux=True),
            (0, 0, 0, 0, 0, 0, None),
        )
        (energies, ((rijs, rkjs), _)), (grad_ra, grad_rc) = vangle_grad(
            ra, rb, rc, theta_0, k, angle_type, box,
        )

    if compute_pressure:
        angle_pressure = jnp.sum(-grad_ra * rijs - grad_rc * rkjs, axis=0)
    else:
        angle_pressure = jnp.zeros(3)
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

    # Safe grad calculation: set angle to 0 only when the beads are collinear
    condition = jnp.logical_and(
        jnp.isclose(cos_phi, 0.0),
        jnp.isclose(sin_phi, 0.0),
    )
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

    # CBT-corrected Martini dihedral always uses the harmonic angle term
    # (GROMACS convention), so call the single-branch helper directly
    # rather than threading a type-1 row through the multi-type dispatch.
    energy_ang, (_, theta) = _angle_potential_harmonic(
        ra, rb, rc, gamma_0, k_phi, box,
    )

    last_angle_energy = jnp.where(
        # fmt: off
        last == 1,
        _angle_potential_harmonic(rb, rc, rd, gamma_0, k_phi, box)[0],
        0.0,
    )
    return energy_dih + energy_ang + last_angle_energy, ((phi, theta), bond_vectors)


def get_dihedral_energy_and_forces(
    forces,
    pos,
    box,
    atom1,
    atom2,
    atom3,
    atom4,
    coeff_or_phase,
    last_or_strength,
    multiplicity=None,
    ff_family="martini",
    compute_pressure=True,
):
    ra = pos[atom1]
    rb = pos[atom2]
    rc = pos[atom3]
    rd = pos[atom4]

    if ff_family == "amber_like":
        phase = coeff_or_phase
        strength = last_or_strength
        multiplicity = jnp.asarray(multiplicity)

        def periodic_torsion_potential(ra, rb, rc, rd, phase, k, n, box):
            phi, bond_vectors = get_dihedral_angle(ra, rb, rc, rd, box)
            energy = k * (1.0 + jnp.cos(n * phi - phase))
            return energy, (phi, bond_vectors)

        dih_grad = vmap(
            value_and_grad(periodic_torsion_potential, (0, 1, 2, 3), has_aux=True),
            (0, 0, 0, 0, 0, 0, 0, None),
        )
        (energies, (phi, (r_ac, r_bc, r_dc))), grads = dih_grad(
            ra,
            rb,
            rc,
            rd,
            phase,
            strength,
            multiplicity,
            box,
        )

        if compute_pressure:
            dihedral_pressure = jnp.sum(-grads[0] * r_ac - grads[1] * r_bc - grads[3] * r_dc, axis=0)
        else:
            dihedral_pressure = jnp.zeros(3)
        forces = forces.at[...].set(0)
        forces = forces.at[atom1].add(-grads[0])
        forces = forces.at[atom2].add(-grads[1])
        forces = forces.at[atom3].add(-grads[2])
        forces = forces.at[atom4].add(-grads[3])
        return jnp.sum(energies), forces, phi, dihedral_pressure

    coeff = coeff_or_phase
    last = last_or_strength

    dih_grad = vmap(
        value_and_grad(cbt_potential, (0, 1, 2, 3), has_aux=True),
        (0, 0, 0, 0, 0, 0, None),
    )
    (energies, (angles, (r_ac, r_bc, r_dc))), grads = dih_grad(
        ra, rb, rc, rd, coeff, last, box
    )

    if compute_pressure:
        dihedral_pressure = jnp.sum(-grads[0] * r_ac - grads[1] * r_bc - grads[3] * r_dc, axis=0)
    else:
        dihedral_pressure = jnp.zeros(3)
    forces = forces.at[...].set(0)
    forces = forces.at[atom1].add(-grads[0])
    forces = forces.at[atom2].add(-grads[1])
    forces = forces.at[atom3].add(-grads[2])
    forces = forces.at[atom4].add(-grads[3])
    return jnp.sum(energies), forces, angles, dihedral_pressure


# Improper dihedrals
def improper_potential(ra, rb, rc, rd, phi_0, k, multiplicity, periodic, box):
    phi, bond_vectors = get_dihedral_angle(ra, rb, rc, rd, box)
    n = multiplicity * 1.0  # ensure float for clean JAX tracing

    periodic_energy = k * (1.0 + jnp.cos(n * phi - phi_0))

    dphi = phi - phi_0
    dphi -= 2 * jnp.pi * jnp.rint(dphi / (2 * jnp.pi))
    harmonic_energy = 0.5 * k * dphi * dphi

    return jnp.where(periodic, periodic_energy, harmonic_energy), bond_vectors


def get_impropers_energy_and_forces(
    forces,
    pos,
    box,
    atom1,
    atom2,
    atom3,
    atom4,
    phi_0,
    k,
    multiplicity=None,
    periodic=None,
    compute_pressure=True,
):
    ra = pos[atom1]
    rb = pos[atom2]
    rc = pos[atom3]
    rd = pos[atom4]

    if multiplicity is None:
        multiplicity = jnp.ones_like(phi_0, dtype=jnp.int32)
    if periodic is None:
        periodic = jnp.zeros_like(phi_0, dtype=bool)

    improper_grad = vmap(
        value_and_grad(improper_potential, (0, 1, 2, 3), has_aux=True),
        (0, 0, 0, 0, 0, 0, 0, 0, None),
    )
    (energies, (r_ac, r_bc, r_dc)), grads = improper_grad(
        ra,
        rb,
        rc,
        rd,
        phi_0,
        k,
        multiplicity,
        periodic,
        box,
    )

    if compute_pressure:
        improper_pressure = jnp.sum(
            -grads[0] * r_ac - grads[1] * r_bc - grads[3] * r_dc, axis=0
        )
    else:
        improper_pressure = jnp.zeros(3)

    forces = forces.at[...].set(0)
    forces = forces.at[atom1].add(-grads[0])
    forces = forces.at[atom2].add(-grads[1])
    forces = forces.at[atom3].add(-grads[2])
    forces = forces.at[atom4].add(-grads[3])
    return jnp.sum(energies), forces, improper_pressure



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
    safe_sin = jnp.maximum(jnp.sin(gamma), 1e-8)
    n_vec = jnp.cross(u_ab, u_cb) / safe_sin
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

"""Alchemical (decoupling) modifications for InteraxFF systems.

InteraxFF systems (``interax.openmm.create_system_from_topology(..., electrostatics="hippo")``) contain:

* a ``NonbondedForce`` with zero charges / epsilons that only carries the Sage intramolecular exceptions,
* electrostatics + polarization in a ``HippoNonbondedForce`` wrapped in a ``CustomCVForce`` (every intramolecular
  pair excluded),
* several intermolecular ``CustomNonbondedForce`` terms (dispersion, exchange, exchange-induction, short-range
  correction), intramolecular pairs excluded,
* a molecular Axilrod-Teller-Muto ``CustomCentroidBondForce`` with one group per molecule and explicit triples,
* the Sage valence terms.

The solute is decoupled from its surroundings, keeping all of its intramolecular interactions:

* electrostatics: ``lambda_electrostatics * E_on + (1 - lambda_electrostatics) * E_off``, where ``E_off`` is a copy of
  the HIPPO force with the solute's charges, multipoles and polarizabilities set to zero. HIPPO has no lambda of its
  own and its energy is not pairwise (mutual / direct polarization), so the two end states are mixed linearly; both
  are evaluated each step (~2x the electrostatics cost). Valid as an alchemical path as long as the sterics stay on
  while it is switched, which is how the absolv protocols order the windows. The two terms are separate
  ``CustomCVForce`` objects (``lambda * E_on`` and ``(1 - lambda) * E_off``), so each HIPPO copy gets its own inner
  Context, which the CUDA and OpenCL platforms require.
* every ``CustomNonbondedForce``: the existing absolv custom-vdW treatment (solute-solvent pairs moved into a copy
  scaled by ``lambda_sterics``). The InteraxFF terms are damped / finite at r -> 0, so no soft core is needed.
* ATM triples that contain a solute molecule move into a copy scaled by ``lambda_sterics``.

Systems converted to the InteraxFF OpenMM plugin (``interaxplugin_convert.convert_system``) carry the dispersion,
exchange, short-range and three-body terms in plugin forces with built-in decoupling: the solute's particles (and its
three-body sites) are flagged alchemical, which scales their pair interactions with everything else by
``lambda_sterics`` and their three-body interactions by ``lambda_electrostatics`` (see ``_plugin_lambda``). The plugin
pair terms stay finite for overlapping atoms (no 0 * inf), which the exported CustomNonbondedForces do not.
"""

import copy

import openmm

from absolv.fep import LAMBDA_ELECTROSTATICS, LAMBDA_STERICS, _add_custom_vdw_lambda


def _clone(force):
    return openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(force))


def is_interax_system(system: openmm.System) -> bool:
    """Whether the system has the InteraxFF layout (HIPPO electrostatics wrapped in a ``CustomCVForce``)."""

    for force in system.getForces():
        if isinstance(force, openmm.CustomCVForce):
            for k in range(force.getNumCollectiveVariables()):
                if isinstance(force.getCollectiveVariable(k), openmm.HippoNonbondedForce):
                    return True
    return False


def _hippo_without(hippo: openmm.HippoNonbondedForce, atoms: set[int]) -> openmm.HippoNonbondedForce:
    """A copy of ``hippo`` in which ``atoms`` carry no charge, multipoles or polarizability."""

    off = _clone(hippo)
    for i in atoms:
        p = list(off.getParticleParameters(i))
        # (charge, dipole, quadrupole, coreCharge, alpha, epsilon, damping, c6, pauliK, pauliQ, pauliAlpha,
        #  polarizability, axisType, atomZ, atomX, atomY)
        p[0] = 0.0
        p[1] = [0.0] * 3
        p[2] = [0.0] * 9
        p[3] = 0.0
        # Not exactly zero (HIPPO's energy is NaN for a zero polarizability), but small enough that the solute does not
        # couple to the solvent at all: a decoupled solute overlaps water atoms, whose core charges are undamped, and
        # with 1e-8 nm^3 the induced-dipole force alpha E grad(E) ~ alpha / r^5 reached 1e7-1e9 kJ/mol/nm at
        # 0.002-0.0002 nm and blew up the lambda = 0 windows (2026-10-23). With 1e-20 the off state equals the solvent
        # alone (to a constant 0.06 kJ/mol, the solute's own term) and the forces stay normal at any overlap.
        p[11] = 1e-20
        off.setParticleParameters(i, *p)
    return off


def _electrostatics_lambda(system: openmm.System, solute_atoms: set[int]):
    """Replace each CustomCVForce that wraps a HippoNonbondedForce by an on/off mix controlled by lambda_electrostatics."""

    for idx in reversed(range(system.getNumForces())):
        force = system.getForce(idx)
        if not isinstance(force, openmm.CustomCVForce):
            continue
        names = [force.getCollectiveVariableName(k) for k in range(force.getNumCollectiveVariables())]
        cvs = [force.getCollectiveVariable(k) for k in range(force.getNumCollectiveVariables())]
        hippo_k = [k for k, cv in enumerate(cvs) if isinstance(cv, openmm.HippoNonbondedForce)]
        if not hippo_k:
            continue
        assert len(cvs) == 1 and force.getEnergyFunction().strip() == names[0], (
            "expected a CustomCVForce whose energy is exactly one HIPPO collective variable"
        )
        hippo = cvs[0]
        # One CustomCVForce per end state, sharing lambda_electrostatics: each CV force evaluates its variables in its
        # own inner Context, and two HippoNonbondedForces in one Context do not compile on CUDA or OpenCL in
        # OpenMM 8.1 (duplicate kernel arguments such as global_coreCharge); only Reference/CPU accept that.
        new_forces = []
        for name, weight, cv in (
            ("E_on", f"{LAMBDA_ELECTROSTATICS}", _clone(hippo)),
            ("E_off", f"(1 - {LAMBDA_ELECTROSTATICS})", _hippo_without(hippo, solute_atoms)),
        ):
            new = openmm.CustomCVForce(f"{weight} * {name}")
            new.addGlobalParameter(LAMBDA_ELECTROSTATICS, 1.0)
            new.addCollectiveVariable(name, cv)
            new.setForceGroup(force.getForceGroup())
            new.setName(f"{force.getName()}:{name}")
            new_forces.append(new)
        # only now: removing the force deletes it, and `hippo` points into it
        system.removeForce(idx)
        for new in new_forces:
            system.addForce(new)


def _plugin_lambda(system: openmm.System, solute_atoms: set[int]):
    """Flag the solute in the InteraxFF plugin forces: their interactions with the alchemical particles scale with
    lambda_sterics (pair forces) or lambda_electrostatics (three-body force). Does nothing for systems without plugin
    forces (or if the plugin is not installed)."""

    try:
        import interaxplugin as ip
    except ImportError:
        return
    for force in system.getForces():
        for cls in (ip.InteraxDispersionForce, ip.InteraxExchangeForce, ip.InteraxShortRangeForce):
            if cls.isinstance(force):
                force = cls.cast(force)
                force.setAlchemicalParameterName(LAMBDA_STERICS)
                for i in solute_atoms:
                    if i < force.getNumParticles():
                        values = list(force.getParticleParameters(i))
                        force.setParticleParameters(i, *values[:-1], True)
        if ip.InteraxThreeBodyForce.isinstance(force):
            # The three-body dispersion diverges (r^-9, of either sign) as molecules meet, while the exchange repulsion
            # that keeps them apart is finite and also scales with lambda_sterics: scaled together, a solute could
            # collapse onto a water at small lambda_sterics. So it is switched off with the electrostatics, while the
            # repulsion is fully on (the FreeSolv schedules turn the electrostatics off first); any path between the
            # end states gives the same free energy.
            force = ip.InteraxThreeBodyForce.cast(force)
            force.setAlchemicalParameterName(LAMBDA_ELECTROSTATICS)
            for k in range(force.getNumSites()):
                particles, alpha, frequency, width, molecule, _ = force.getSiteParameters(k)
                if solute_atoms.intersection(particles):
                    force.setSiteParameters(k, particles, alpha, frequency, width, molecule, True)


def _atm_lambda(system: openmm.System, solute_atoms: set[int]):
    """Split each molecular ATM CustomCentroidBondForce into solute-free triples and lambda_sterics-scaled solute triples."""

    for idx in reversed(range(system.getNumForces())):
        force = system.getForce(idx)
        if not isinstance(force, openmm.CustomCentroidBondForce):
            continue
        n_groups = force.getNumGroups()
        groups = [force.getGroupParameters(g) for g in range(n_groups)]
        solute_groups = {g for g, (atoms, _) in enumerate(groups) if solute_atoms.intersection(atoms)}

        expression = force.getEnergyFunction().split(";")
        scaled = list(expression)
        scaled[0] = f"{LAMBDA_STERICS}*({expression[0]})"

        def build(energy):
            f = openmm.CustomCentroidBondForce(force.getNumGroupsPerBond(), energy)
            for k in range(force.getNumPerBondParameters()):
                f.addPerBondParameter(force.getPerBondParameterName(k))
            for k in range(force.getNumGlobalParameters()):
                f.addGlobalParameter(force.getGlobalParameterName(k), force.getGlobalParameterDefaultValue(k))
            for atoms, weights in groups:
                f.addGroup(list(atoms), list(weights))
            f.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
            f.setForceGroup(force.getForceGroup())
            f.setName(force.getName())
            return f

        plain, alch = build(force.getEnergyFunction()), build(";".join(scaled))
        alch.addGlobalParameter(LAMBDA_STERICS, 1.0)
        for b in range(force.getNumBonds()):
            g, params = force.getBondParameters(b)
            (alch if solute_groups.intersection(g) else plain).addBond(list(g), list(params))
        system.removeForce(idx)
        system.addForce(plain)
        if alch.getNumBonds() > 0:
            system.addForce(alch)


def apply_fep_interax(
    system: openmm.System,
    alchemical_indices: list[set[int]],
    persistent_indices: list[set[int]],
) -> openmm.System:
    """Return a copy of an InteraxFF system in which the alchemical molecules can be decoupled.

    Args:
        system: The InteraxFF system (``electrostatics="hippo"``).
        alchemical_indices: Atom indices of each alchemical (solute) molecule.
        persistent_indices: Atom indices of each persistent (solvent) molecule.
    """

    system = copy.deepcopy(system)
    solute_atoms = {i for values in alchemical_indices for i in values}

    # Lone-pair virtual sites (interax.lonepairs) are extra particles appended after all atoms; a site whose
    # defining particles belong to a solute molecule carries that molecule's q_lp and must be decoupled with it,
    # or the "decoupled" solute keeps pulling on the solvent through its sites (found 2026-10-23: trimethylamine
    # lambda=0 end state off by +2.7 kJ/mol with models_v3R4bfdE1La).
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            vs = system.getVirtualSite(i)
            parents = {vs.getParticle(k) for k in range(vs.getNumParticles())}
            if parents & solute_atoms:
                solute_atoms.add(i)

    _electrostatics_lambda(system, solute_atoms)

    custom = [f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce)]
    for force in custom:
        aa_na, aa_aa = _add_custom_vdw_lambda(force, alchemical_indices, persistent_indices, None)
        aa_na.setForceGroup(force.getForceGroup())
        system.addForce(aa_na)
        # Intramolecular solute pairs are all excluded in InteraxFF forces, so the alchemical-alchemical copy would
        # be empty; keep it only if the solute has non-excluded intramolecular pairs.
        excluded = {tuple(sorted(aa_aa.getExclusionParticles(k))) for k in range(aa_aa.getNumExclusions())}
        pairs = {(i, j) for mol in alchemical_indices for i in mol for j in mol if i < j}
        if pairs - excluded:
            aa_aa.setForceGroup(force.getForceGroup())
            system.addForce(aa_aa)

    _atm_lambda(system, solute_atoms)
    _plugin_lambda(system, solute_atoms)
    return system

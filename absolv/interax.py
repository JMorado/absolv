"""Prepare InteraxFF systems for FEP calculations.

InteraxFF systems (``interax.openmm.create_system_from_topology`` with
``electrostatics="hippo"``) contain:

* a ``NonbondedForce`` with zero charges and epsilons that only carries the
  intramolecular exceptions of the base force field,
* electrostatics and polarization in a ``HippoNonbondedForce`` wrapped in a
  ``CustomCVForce``, with every intramolecular pair excluded,
* the intermolecular dispersion, exchange repulsion and short-range terms, either
  as ``CustomNonbondedForce`` objects or as InteraxFF OpenMM plugin forces,
* a molecular Axilrod-Teller-Muto term, either as a ``CustomCentroidBondForce`` or
  as an ``InteraxThreeBodyForce``,
* the valence terms of the base force field.

The solute is decoupled from its surroundings, keeping all of its intramolecular
interactions:

* electrostatics: ``lambda_electrostatics * E_on + (1 - lambda_electrostatics) *
  E_off``, where ``E_off`` is a copy of the HIPPO force in which the solute carries
  no charges, multipoles or polarizability. HIPPO has no lambda of its own and its
  polarization energy is not pairwise, so the two end states are mixed linearly.
  This is a valid path as long as the sterics are fully on while it is switched,
  which is how the absolv protocols order the windows.
* the pair terms are scaled by ``lambda_sterics``. They are finite as two atoms
  meet, so no soft core is needed.
* the three-body terms that involve the solute are scaled by ``lambda_sterics``
  (``CustomCentroidBondForce``) or ``lambda_electrostatics``
  (``InteraxThreeBodyForce``, see ``_add_plugin_lambda``).
"""

import copy

import openmm

from absolv.fep import LAMBDA_ELECTROSTATICS, LAMBDA_STERICS, _add_custom_vdw_lambda


def _clone(force: openmm.Force) -> openmm.Force:
    return openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(force))


def _wrapped_hippo(force: openmm.Force) -> openmm.HippoNonbondedForce | None:
    """Returns the HIPPO force wrapped by a ``CustomCVForce``, or ``None``."""

    if not isinstance(force, openmm.CustomCVForce):
        return None

    for i in range(force.getNumCollectiveVariables()):
        if isinstance(force.getCollectiveVariable(i), openmm.HippoNonbondedForce):
            return force.getCollectiveVariable(i)

    return None


def is_interax_system(system: openmm.System) -> bool:
    """Returns whether a system has the InteraxFF layout, i.e. HIPPO electrostatics
    wrapped in a ``CustomCVForce``."""

    return any(_wrapped_hippo(force) is not None for force in system.getForces())


def _hippo_without(
    hippo: openmm.HippoNonbondedForce, atom_indices: set[int]
) -> openmm.HippoNonbondedForce:
    """Returns a copy of a HIPPO force in which some atoms carry no charges,
    multipoles or polarizability.

    Args:
        hippo: The HIPPO force to copy.
        atom_indices: The indices of the atoms to remove the interactions of.
    """

    hippo_off = _clone(hippo)

    for i in atom_indices:
        # charge, dipole, quadrupole, core charge, ..., polarizability (index 11), ...
        parameters = list(hippo_off.getParticleParameters(i))
        parameters[0] = 0.0
        parameters[1] = [0.0] * 3
        parameters[2] = [0.0] * 9
        parameters[3] = 0.0
        # A zero polarizability makes the HIPPO energy NaN. It must still be tiny, as
        # a decoupled solute overlaps water atoms whose core charges are undamped:
        # 1e-8 nm^3 gave forces of up to 1e9 kJ/mol/nm and NaN windows.
        parameters[11] = 1.0e-20
        hippo_off.setParticleParameters(i, *parameters)

    return hippo_off


def _add_electrostatics_lambda(system: openmm.System, solute_indices: set[int]):
    """Replaces each ``CustomCVForce`` that wraps a HIPPO force with a mix of the
    HIPPO force and a copy without the solute, controlled by
    ``lambda_electrostatics``.

    Args:
        system: The system to modify in place.
        solute_indices: The indices of the solute particles.
    """

    for force_idx in reversed(range(system.getNumForces())):
        force = system.getForce(force_idx)

        hippo = _wrapped_hippo(force)

        if hippo is None:
            continue

        assert (
            force.getNumCollectiveVariables() == 1
            and force.getEnergyFunction().strip() == force.getCollectiveVariableName(0)
        ), "the HIPPO force should be the only collective variable"

        # Each end state gets its own CustomCVForce, i.e. its own inner context, as
        # two HIPPO forces cannot be compiled into one CUDA or OpenCL context.
        end_states = [
            ("E_on", LAMBDA_ELECTROSTATICS, _clone(hippo)),
            (
                "E_off",
                f"(1 - {LAMBDA_ELECTROSTATICS})",
                _hippo_without(hippo, solute_indices),
            ),
        ]
        new_forces = []

        for name, weight, hippo_state in end_states:
            new_force = openmm.CustomCVForce(f"{weight} * {name}")
            new_force.addGlobalParameter(LAMBDA_ELECTROSTATICS, 1.0)
            new_force.addCollectiveVariable(name, hippo_state)
            new_force.setForceGroup(force.getForceGroup())
            new_force.setName(f"{force.getName()}:{name}")
            new_forces.append(new_force)

        # Removing the force deletes ``hippo``, so this must come after the copies.
        system.removeForce(force_idx)

        for new_force in new_forces:
            system.addForce(new_force)


def _copy_centroid_force(
    force: openmm.CustomCentroidBondForce, energy: str
) -> openmm.CustomCentroidBondForce:
    """Returns a copy of a ``CustomCentroidBondForce`` with a new energy expression,
    with the same parameters and groups but no bonds.

    Args:
        force: The force to copy.
        energy: The energy expression of the copy.
    """

    new_force = openmm.CustomCentroidBondForce(force.getNumGroupsPerBond(), energy)

    for i in range(force.getNumPerBondParameters()):
        new_force.addPerBondParameter(force.getPerBondParameterName(i))
    for i in range(force.getNumGlobalParameters()):
        new_force.addGlobalParameter(
            force.getGlobalParameterName(i), force.getGlobalParameterDefaultValue(i)
        )
    for i in range(force.getNumGroups()):
        atoms, weights = force.getGroupParameters(i)
        new_force.addGroup(list(atoms), list(weights))

    new_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
    new_force.setForceGroup(force.getForceGroup())
    new_force.setName(force.getName())
    return new_force


def _add_atm_lambda(system: openmm.System, solute_indices: set[int]):
    """Splits each three-body ``CustomCentroidBondForce`` into the triples without the
    solute and those with it, which are scaled by ``lambda_sterics``.

    Args:
        system: The system to modify in place.
        solute_indices: The indices of the solute particles.
    """

    for force_idx in reversed(range(system.getNumForces())):
        force = system.getForce(force_idx)

        if not isinstance(force, openmm.CustomCentroidBondForce):
            continue

        groups = [force.getGroupParameters(i) for i in range(force.getNumGroups())]
        solute_groups = {
            i
            for i, (atoms, _) in enumerate(groups)
            if solute_indices.intersection(atoms)
        }

        energy_expression = force.getEnergyFunction().split(";")
        energy_expression[0] = f"{LAMBDA_STERICS}*({energy_expression[0]})"

        chemical_force = _copy_centroid_force(force, force.getEnergyFunction())
        alchemical_force = _copy_centroid_force(force, ";".join(energy_expression))
        alchemical_force.addGlobalParameter(LAMBDA_STERICS, 1.0)

        for i in range(force.getNumBonds()):
            bond_groups, parameters = force.getBondParameters(i)
            target = (
                alchemical_force
                if solute_groups.intersection(bond_groups)
                else chemical_force
            )
            target.addBond(list(bond_groups), list(parameters))

        system.removeForce(force_idx)
        system.addForce(chemical_force)

        if alchemical_force.getNumBonds() > 0:
            system.addForce(alchemical_force)


def _add_plugin_lambda(system: openmm.System, solute_indices: set[int]):
    """Flags the solute as alchemical in the InteraxFF OpenMM plugin forces.

    Notes:
        * The pair forces scale the interactions of the solute with everything else
          by ``lambda_sterics``.
        * The three-body force is scaled by ``lambda_electrostatics`` instead. It
          diverges as molecules meet, while the exchange repulsion that keeps them
          apart is finite, so it is switched off while the repulsion is fully on.

    Args:
        system: The system to modify in place. Nothing is done if it has no plugin
            forces or the plugin is not installed.
        solute_indices: The indices of the solute particles.
    """

    try:
        import interaxplugin
    except ImportError:
        return

    pair_force_types = [
        interaxplugin.InteraxDispersionForce,
        interaxplugin.InteraxExchangeForce,
        interaxplugin.InteraxShortRangeForce,
    ]

    for force in system.getForces():
        for force_type in pair_force_types:
            if not force_type.isinstance(force):
                continue

            pair_force = force_type.cast(force)
            pair_force.setAlchemicalParameterName(LAMBDA_STERICS)

            for i in solute_indices:
                # the parameters end with the alchemical flag
                *parameters, _ = pair_force.getParticleParameters(i)
                pair_force.setParticleParameters(i, *parameters, True)

        if interaxplugin.InteraxThreeBodyForce.isinstance(force):
            three_body_force = interaxplugin.InteraxThreeBodyForce.cast(force)
            three_body_force.setAlchemicalParameterName(LAMBDA_ELECTROSTATICS)

            for i in range(three_body_force.getNumSites()):
                *parameters, _ = three_body_force.getSiteParameters(i)

                if solute_indices.intersection(parameters[0]):
                    three_body_force.setSiteParameters(i, *parameters, True)


def apply_fep_interax(
    system: openmm.System,
    alchemical_indices: list[set[int]],
    persistent_indices: list[set[int]],
) -> openmm.System:
    """Generate an InteraxFF system whereby a number of the molecules can be
    decoupled from the rest.

    Args:
        system: The InteraxFF system (``electrostatics="hippo"``).
        alchemical_indices: The atom indices corresponding to each molecule that
            should be decoupled.
        persistent_indices: The atom indices corresponding to each molecule that
            should **not** be decoupled.

    Returns:
        The alchemical system.
    """

    system = copy.deepcopy(system)

    solute_indices = {i for indices in alchemical_indices for i in indices}

    # Lone-pair virtual sites follow all the atoms; a site of a solute molecule
    # carries part of its charge and must be decoupled with it.
    for i in range(system.getNumParticles()):
        if not system.isVirtualSite(i):
            continue

        virtual_site = system.getVirtualSite(i)
        parents = {
            virtual_site.getParticle(j) for j in range(virtual_site.getNumParticles())
        }

        if parents & solute_indices:
            solute_indices.add(i)

    _add_electrostatics_lambda(system, solute_indices)

    custom_forces = [
        force
        for force in system.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
    ]

    for force in custom_forces:
        aa_na_force, aa_aa_force = _add_custom_vdw_lambda(
            force, alchemical_indices, persistent_indices, None
        )
        aa_na_force.setForceGroup(force.getForceGroup())
        system.addForce(aa_na_force)

        # InteraxFF excludes every intramolecular pair, so the alchemical-alchemical
        # force is only kept if a solute has pairs that are not excluded.
        excluded_pairs = {
            tuple(sorted(aa_aa_force.getExclusionParticles(i)))
            for i in range(aa_aa_force.getNumExclusions())
        }
        solute_pairs = {
            (i, j)
            for indices in alchemical_indices
            for i in indices
            for j in indices
            if i < j
        }

        if solute_pairs - excluded_pairs:
            aa_aa_force.setForceGroup(force.getForceGroup())
            system.addForce(aa_aa_force)

    _add_atm_lambda(system, solute_indices)
    _add_plugin_lambda(system, solute_indices)

    return system

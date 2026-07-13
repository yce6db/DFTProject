"""
Generate a dimer vs. bulk-solid-analogue comparison figure,
in the style of the uploaded reference image.

Change from the earlier per-system scripts: every ionic pair (NaCl,
LiF, MgO, MgS, NaF, LiCl, AlP, BP) is now built as a single diatomic
monomer (2 atoms) -- the same convention already used for LiCl and
BN/SiC/BP -- rather than the 4-atom (MX)2 rhombus cluster built in
earlier scripts. Covalent diamond-analogues (ethane, disilane) are
left as-is, since they were never doubled in the first place.

Diatomic bond lengths are estimated as the sum of covalent radii
(ase.data.covalent_radii). This is a reasonable geometric placeholder,
NOT a literature-quality bond length -- relax each structure at your
level of theory before using it for production comparisons.

Solid-side lattice constants are standard literature values for the
respective bulk structure (rocksalt, zincblende, or diamond).
"""

import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import bulk, molecule
from ase.data import covalent_radii, atomic_numbers
from ase.visualize.plot import plot_atoms


def diatomic(sym1, sym2, vacuum=6.0):
    """Simple 2-atom molecule; bond length = sum of covalent radii."""
    d = covalent_radii[atomic_numbers[sym1]] + covalent_radii[atomic_numbers[sym2]]
    atoms = Atoms(sym1 + sym2, positions=[[0, 0, 0], [0, 0, d]])
    atoms.center(vacuum=vacuum)
    atoms.pbc = False
    return atoms


def rocksalt_cluster(sym1, sym2, a):
    return bulk(sym1 + sym2, crystalstructure='rocksalt', a=a, cubic=True)


def zincblende_cluster(sym1, sym2, a):
    return bulk(sym1 + sym2, crystalstructure='zincblende', a=a, cubic=True)


def diamond_cluster(sym, a):
    return bulk(sym, crystalstructure='diamond', a=a, cubic=True)


# ---- Define all systems: (molecule title, molecule Atoms, solid title, solid Atoms) ----
systems = []

# Zincblende-structure covalent semiconductors
systems.append(('AlP dimer', diatomic('Al', 'P'), 'AlP', zincblende_cluster('Al', 'P', 5.458)))
systems.append(('BP dimer',  diatomic('B',  'P'), 'BP',  zincblende_cluster('B',  'P', 4.538)))
systems.append(('BN dimer',  diatomic('B',  'N'), 'BN',  zincblende_cluster('B',  'N', 3.607)))
systems.append(('SiC dimer', diatomic('Si', 'C'), 'SiC', zincblende_cluster('Si', 'C', 4.358)))

# Diamond-structure elemental analogues (already single small molecules)
ethane = molecule('C2H6')
ethane.center(vacuum=6.0)
ethane.pbc = False
systems.append(('Ethane', ethane, 'C', diamond_cluster('C', 3.567)))

disilane = molecule('Si2H6')
disilane.center(vacuum=6.0)
disilane.pbc = False
systems.append(('Disilane', disilane, 'Si', diamond_cluster('Si', 5.43)))

# Rocksalt-structure ionic compounds -- simple diatomics, matching the LiCl convention
systems.append(('LiCl dimer', diatomic('Li', 'Cl'), 'LiCl', rocksalt_cluster('Li', 'Cl', 5.106)))
systems.append(('LiF dimer',  diatomic('Li', 'F'),  'LiF',  rocksalt_cluster('Li', 'F', 4.01)))
systems.append(('NaCl dimer', diatomic('Na', 'Cl'), 'NaCl', rocksalt_cluster('Na', 'Cl', 5.595)))
systems.append(('NaF dimer',  diatomic('Na', 'F'),  'NaF',  rocksalt_cluster('Na', 'F', 4.609)))
systems.append(('MgO dimer',  diatomic('Mg', 'O'),  'MgO',  rocksalt_cluster('Mg', 'O', 4.207)))
systems.append(('MgS dimer',  diatomic('Mg', 'S'),  'MgS',  rocksalt_cluster('Mg', 'S', 5.202)))

# ---- Build the comparison grid ----
n = len(systems)
fig, axes = plt.subplots(n, 2, figsize=(6, 3.0 * n))

for i, (mol_title, mol_atoms, solid_title, solid_atoms) in enumerate(systems):
    ax_mol, ax_solid = axes[i, 0], axes[i, 1]

    plot_atoms(mol_atoms, ax_mol, radii=0.5, rotation='10x,10y,0z')
    ax_mol.set_title(mol_title, fontsize=10)
    ax_mol.set_axis_off()

    plot_atoms(solid_atoms, ax_solid, radii=0.5, rotation='10x,10y,0z', show_unit_cell=2)
    ax_solid.set_title(solid_title, fontsize=10)
    ax_solid.set_axis_off()

axes[0, 0].annotate('Dimer', xy=(0.5, 1.25), xycoords='axes fraction',
                     ha='center', fontsize=13, fontweight='bold')
axes[0, 1].annotate('Bulk solid analogue', xy=(0.5, 1.25), xycoords='axes fraction',
                     ha='center', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('dimer_solid_comparison.png', dpi=150, bbox_inches='tight')
print('done')

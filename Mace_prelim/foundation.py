import pandas as pd
import numpy as np
from ase.build import bulk
from ase.eos import EquationOfState
from ase.units import kJ 
from ase import Atoms
from mace.calculators import mace_mp
from collections import Counter
import matplotlib.pyplot as plt


experiment = pd.read_csv("Mace_prelim/exp.csv")
calc = mace_mp(model="medium",dispersion = False, default_dtype = "float64", device = "cpu")

def calculate_bulk_and_lattice_constant(element,structure,lattice_prediction):
    lat_values = np.linspace(lattice_prediction-0.2,lattice_prediction+0.2,15)
    volumes = []
    energies = []
    for lat in lat_values:
        atoms = bulk(element, structure, a=lat)
        atoms.set_calculator(calc)
        energy = atoms.get_potential_energy()
        volume = atoms.get_volume()
        volumes.append(volume)
        energies.append(energy)
    eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
    v0, e0, B = eos.fit()
    if structure == "fcc":
        a0 = (4*v0)**(1/3)
    elif structure == "bcc":
        a0 = (2*v0)**(1/3)
    elif structure == "rocksalt":
        a0 = (4*v0)**(1/3)
    elif structure == "diamond":
        a0 = (4*v0)**(1/3)
    elif structure == "zincblende":
        a0 = (4*v0)**(1/3)      
   
    B_GPa = B / kJ * 1e24
    # Plot EOS, annotate with element name, and save
    eos.plot()  # draws on current matplotlib axes
    ax = plt.gca()
    ax.text(
        0.05,
        0.95,
        element,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )
    plt.savefig(f"{element}_.png", bbox_inches="tight")
    plt.close()
    return a0, B_GPa

def calculate_cohesive_energy(element,structure,lattice_constant):
    atoms_bulk = bulk(element, structure, a=lattice_constant)
    atoms_bulk.set_calculator(calc)
    energy_bulk = atoms_bulk.get_potential_energy()
    if len(Counter(atoms_bulk.get_chemical_symbols())) == 1:
        atom_individual = Atoms(element, positions=[[0, 0, 0]],pbc=False)
        atom_individual.set_calculator(calc)
        energy_individual = atom_individual.get_potential_energy()
        cohesive_energy = -(energy_individual - (energy_bulk / len(atoms_bulk)))
    else:
        energy_individual = 0
        for symbol in set(atoms_bulk.get_chemical_symbols()):
            atom_individual = Atoms(symbol, positions=[[0, 0, 0]],pbc=False)
            atom_individual.set_calculator(calc)
            energy_individual += atom_individual.get_potential_energy()
        cohesive_energy = -(energy_individual - energy_bulk)/ len(atoms_bulk)
    return cohesive_energy
calculate_bulk_and_lattice_constant("Li", "bcc", 3.477)
#with open("results.txt", "w") as f:
    #f.write(f"{'Element':<10}{'Structure':<12}{'Lattice Å':>12}{'Bulk GPa':>12}{'Cohesive eV':>15}\n")
    #f.write("-" * 61 + "\n")
    #for index, row in experiment.iterrows():
        #element = row["Solid"]
       #structure = row["Structures"]
        #lattice_prediction = row["Lattice Constant (Å)"]
        #a0, B_GPa = calculate_bulk_and_lattice_constant(element, structure, lattice_prediction)
        #cohesive_energy = calculate_cohesive_energy(element, structure, a0)
        
        #f.write(
            #f"{element:<10}{structure:<12}"
            #f"{a0:12.4f}{B_GPa:12.2f}{cohesive_energy:15.4f}\n"
        #)
    
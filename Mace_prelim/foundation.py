import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ase.build import bulk
from ase.eos import EquationOfState
from ase.units import kJ 
from ase import Atoms
from mace.calculators import mace_mp
from mace.calculators.mace import MACECalculator
from collections import Counter


SCRIPT_DIR = Path(__file__).resolve().parent
experiment = pd.read_csv(SCRIPT_DIR / "exp.csv")
#MACE_MODEL_PATH = SCRIPT_DIR / "mace-omat-0-medium.model"
#calc = MACECalculator(model_path=MACE_MODEL_PATH, device="cpu", default_dtype="float64", dispersion=False)
# calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device="cpu")

CALCULATOR_LIST = {
    "MACE-OMAT-0": MACECalculator(model_path=str(SCRIPT_DIR / "mace-omat-0-medium.model"), device="cpu", default_dtype="float64", dispersion=False),
    "MACE-MP-0": mace_mp(model="medium", dispersion=False, default_dtype="float64", device="cpu"),
    "MACE-MPA-0": MACECalculator(model_path=str(SCRIPT_DIR / "mace-mpa-0-medium.model"), device="cpu", default_dtype="float64", dispersion=False),
    "MACE-MATPES-r2SCAN-0": MACECalculator(model_path=str(SCRIPT_DIR / "mace-matpes-r2scan-omat-ft.model"), device="cpu", default_dtype="float64", dispersion=False),
}

PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def calculate_bulk_and_lattice_constant(element, structure, lattice_prediction, calculator):
    lat_values = np.linspace(lattice_prediction - 0.2, lattice_prediction + 0.2, 15)
    volumes = []
    energies = []
    for lat in lat_values:
        atoms = bulk(element, structure, a=lat)
        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy()
        volumes.append(atoms.get_volume())
        energies.append(energy)
    eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
    v0, e0, B = eos.fit()
    if structure == "fcc":
        a0 = (4 * v0) ** (1 / 3)
    elif structure == "bcc":
        a0 = (2 * v0) ** (1 / 3)
    elif structure == "rocksalt":
        a0 = (4 * v0) ** (1 / 3)
    elif structure == "diamond":
        a0 = (4 * v0) ** (1 / 3)
    elif structure == "zincblende":
        a0 = (4 * v0) ** (1 / 3)
    else:
        raise ValueError(f"Unsupported structure: {structure}")

    B_GPa = B / kJ * 1e24
    return a0, B_GPa, lat_values, volumes, energies

def plot_eos_curves_for_solid(element, structure, lattice_prediction, calculators):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (label, calculator) in zip(axes, calculators.items()):
        a0, B_GPa, lat_values, volumes, energies = calculate_bulk_and_lattice_constant(
            element, structure, lattice_prediction, calculator
        )

        eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
        v0, e0, B = eos.fit()

        ax.plot(volumes, energies, "o")

        v_fit = np.linspace(min(volumes), max(volumes), 100)
        e_fit = eos.func(v_fit, *eos.eos_parameters)

        ax.plot(v_fit, e_fit, "-")

        ax.set_title(label)
        ax.set_xlabel("Volume (Å³)")
        ax.set_ylabel("Energy (eV)")


        ax.legend(fontsize=8)

    fig.suptitle(f"EOS comparison for {element} ({structure})", fontsize=14)
    fig.tight_layout()

    output_file = PLOTS_DIR / f"{element}_{structure}_eos_subplots.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

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


rows = []
for index, row in experiment.iterrows():
    element = row["Solid"]
    structure = row["Structures"]
    lattice_prediction = row["Lattice Constant (Å)"]
    
   
    plot_eos_curves_for_solid(element, structure, lattice_prediction, CALCULATOR_LIST)
    
    
    #for calc_name, calculator in CALCULATOR_LIST.items():
        #a0, B_GPa, _, _, _ = calculate_bulk_and_lattice_constant(element, structure, lattice_prediction, calculator)
        #rows.append({
            #"Element": element,
            #"Structure": structure,
            #"Calculator": calc_name,
           # "Lattice Å": a0,
            #"Bulk GPa": B_GPa,
            #"Cohesive eV": cohesive_energy,
       # })

#df = pd.DataFrame(rows)
#df.to_csv("results.csv", index=False, encoding="utf-8")
    
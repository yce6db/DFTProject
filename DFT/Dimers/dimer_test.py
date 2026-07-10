from ase.build import molecule
from ase.visualize import view
from ase.io import write

# ---- 1. Build the molecule ----
si2h6 = molecule('Si2H6')  # ASE's built-in reference geometry (staggered, G2 test set)

si2h6.center(vacuum=10.0)  # pad with vacuum so it can be treated as isolated
si2h6.pbc = False            # non-periodic

print(f"Si-Si bond length: {si2h6.get_distance(0, 1):.3f} Angstrom")
print(f"Cell: {si2h6.cell}")

# ---- 2. Save structure to disk ----
si2h6.write('disilane.xyz')

# ---- 3. Interactive viewer (opens ASE's built-in GUI) ----
# Comment this out if running non-interactively / on a headless machine.
view(si2h6)

# ---- 4. Static image, saved to file (works headless, no GUI needed) ----
write('disilane.png', si2h6, rotation='30x,30y,0z')
from ase.io import read

atoms = read("/path/to/your/Y.extxyz", index=-1)
print("Number of atoms:", len(atoms))
print("Cell:", atoms.cell)
print("PBC:", atoms.pbc)

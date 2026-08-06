"""
Create a compact grid figure showing all 12 bulk solids
(III-V, diamond-type, and rock-salt structures).

Required packages:
    pip install ase matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec

from ase.io import read
from ase.visualize.plot import plot_atoms


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DFT_CALC_DIR = Path("/mnt/ceph/users/cwoodson/dft_calc")
BULK_DIR = DFT_CALC_DIR / "latbulk" / "eos_jobs"

OUTPUT_FILE = SCRIPT_DIR / "representative_structures.png"

DPI = 200

# All 12 bulk solids: (title, label, system_name).
BULK_STRUCTURES = [
    ("III–V solid", "AlP", "AlP"),
    ("III–V solid", "BN", "BN"),
    ("III–V solid", "BP", "BP"),
    ("Diamond-type solid", "C", "C"),
    ("Diamond-type solid", "Si", "Si"),
    ("Diamond-type solid", "SiC", "SiC"),
    ("Rock-salt solid", "NaF", "NaF"),
    ("Rock-salt solid", "NaCl", "NaCl"),
    ("Rock-salt solid", "LiCl", "LiCl"),
    ("Rock-salt solid", "LiF", "LiF"),
    ("Rock-salt solid", "MgO", "MgO"),
    ("Rock-salt solid", "MgS", "MgS"),
]

BULK_ROTATION = "10x,20y,0z"
BULK_RADII = 0.78
BULK_SCALE = 0.90

# Sized for direct placement on a poster/slide.
TITLE_FONT_SIZE = 22
LABEL_FONT_SIZE = 18
MISSING_FONT_SIZE = 16

GRID_ROWS = 3
GRID_COLS = 4


# =============================================================================
# File-location functions
# =============================================================================

def first_existing_path(candidates):
    """
    Return the first existing file from a list of candidate paths.
    """

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def find_bulk_structure(system_name):
    """
    Locate a bulk-solid geometry.
    """

    system_dir = BULK_DIR / system_name

    candidates = [
        system_dir / "scale_1.000" / "geometry.in",
        system_dir / "geometry.in",
        system_dir / "eos_jobs" / "scale_1.000" / "geometry.in",
        BULK_DIR / f"{system_name}.cif",
        BULK_DIR / f"{system_name}.xyz",
    ]

    structure_file = first_existing_path(candidates)

    if structure_file is not None:
        return structure_file

    if system_dir.is_dir():
        geometry_files = sorted(system_dir.rglob("geometry.in"))

        for geometry_file in geometry_files:
            if geometry_file.parent.name == "scale_1.000":
                return geometry_file

        if geometry_files:
            return geometry_files[0]

    return None


# =============================================================================
# Structure reading and preparation
# =============================================================================

def read_structure(structure_file):
    """
    Read an ASE-supported structure file.
    """

    if structure_file is None:
        return None

    try:
        if structure_file.name == "aims.out":
            return read(
                structure_file,
                index=-1,
                format="aims-output",
            )

        return read(
            structure_file,
            index=-1,
        )

    except Exception as error:
        print(f"[WARN] Could not read {structure_file}: {error}")
        return None


def prepare_periodic_structure(atoms):
    """
    Prepare a periodic structure for plotting.
    """

    atoms = atoms.copy()

    cell_lengths = atoms.cell.lengths()

    if all(length > 0.0 for length in cell_lengths):
        atoms.set_pbc(True)
        atoms.wrap()

    return atoms


# =============================================================================
# Plotting helpers
# =============================================================================

def clear_axis(axis):
    """
    Remove standard Matplotlib ticks, labels, and spines.
    """

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")

    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.patch.set_alpha(0.0)


def show_missing_structure(axis, title, label):
    """
    Display a warning when a structure file cannot be found or read.
    """

    clear_axis(axis)

    axis.text(
        0.5,
        0.50,
        "Structure\nnot found",
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=MISSING_FONT_SIZE,
        fontweight="bold",
        color="crimson",
    )

    axis.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=16,
    )

    axis.text(
        0.5,
        -0.12,
        label,
        ha="center",
        va="top",
        transform=axis.transAxes,
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
    )


def plot_structure(axis, atoms, title, label, rotation, radii, scale):
    """
    Plot one ASE Atoms object with a bold title and a bold label below it.
    """

    clear_axis(axis)

    if atoms is None:
        show_missing_structure(axis, title, label)
        return

    try:
        plot_atoms(
            atoms,
            axis,
            rotation=rotation,
            radii=radii,
            scale=scale,
            show_unit_cell=2,
        )

        axis.set_title(
            title,
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
            pad=16,
        )

        axis.text(
            0.5,
            -0.12,
            label,
            ha="center",
            va="top",
            transform=axis.transAxes,
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
        )

        axis.set_aspect("equal")
        clear_axis(axis)

    except Exception as error:
        print(f"[WARN] Could not plot {title}: {error}")
        show_missing_structure(axis, title, label)


def load_and_plot(axis, title, label, system_name):
    """
    Locate, read, and plot one bulk structure.
    """

    structure_file = find_bulk_structure(system_name)
    atoms = read_structure(structure_file)

    if atoms is not None:
        atoms = prepare_periodic_structure(atoms)

    plot_structure(
        axis=axis,
        atoms=atoms,
        title=title,
        label=label,
        rotation=BULK_ROTATION,
        radii=BULK_RADII,
        scale=BULK_SCALE,
    )

    print(f"bulk {system_name:>10}: {structure_file}")


# =============================================================================
# Main routine
# =============================================================================

def main():
    print(f"Running script: {Path(__file__).resolve()}")
    print()

    figure = plt.figure(figsize=(20, 14))
    figure.patch.set_alpha(0.0)

    grid = gridspec.GridSpec(
        GRID_ROWS,
        GRID_COLS,
        figure=figure,
        wspace=0.35,
        hspace=0.45,
    )

    for index, (title, label, system_name) in enumerate(BULK_STRUCTURES):
        row = index // GRID_COLS
        column = index % GRID_COLS

        axis = figure.add_subplot(grid[row, column])

        load_and_plot(
            axis=axis,
            title=title,
            label=label,
            system_name=system_name,
        )

    figure.tight_layout(pad=2.5, w_pad=2.0, h_pad=2.5)

    figure.savefig(
        OUTPUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
        transparent=True,
    )

    print()
    print(f"Figure written to: {OUTPUT_FILE.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()


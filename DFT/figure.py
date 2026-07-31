"""
Create a compact 2x4 grid figure with:

    Row 1: One dimer from each category (III-V, diamond-type, rock-salt)
    Row 2: One bulk solid from each category (III-V, diamond-type, rock-salt)
    Right column (both rows): One ROY polymorph

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
XYZ_ROOT = DFT_CALC_DIR
EXTXYZ_ROOT = SCRIPT_DIR

DIMER_DIR = DFT_CALC_DIR / "diatomics" / "dimer_jobs"
BULK_DIR = DFT_CALC_DIR / "latbulk" / "eos_jobs"

OUTPUT_FILE = SCRIPT_DIR / "representative_structures.png"

DPI = 200

# Top row: one dimer per category.
DIMER_STRUCTURES = [
    ("III–V dimer", "BH$_3$NH$_3$", "BH3NH3"),
    ("Diamond-type dimer", "Disilane", "Si2H6"),
    ("Rock-salt dimer", "NaCl dimer", "NaCl"),
]

# Bottom row: one bulk solid per category.
BULK_STRUCTURES = [
    ("III–V solid", "BN", "BN"),
    ("Diamond-type solid", "Si", "Si"),
    ("Rock-salt solid", "NaCl", "NaCl"),
]

# Right column: one ROY polymorph, spanning both rows.
# NOTE: the underlying file is named "Y_fixed.extxyz" in DFT, but the
# displayed label stays "Y" since that's the polymorph name.
ROY_STRUCTURE = ("ROY polymorph", "Y", "Y_fixed")

DIMER_ROTATION = "10x,20y,0z"
BULK_ROTATION = "10x,20y,0z"
# Default (no rotation) so the ROY polymorph renders the same way
# ASE GUI shows it by default (looking straight down the z-axis).
ROY_ROTATION = "0x,0y,0z"

DIMER_RADII = 0.75
BULK_RADII = 0.78
# Default ASE covalent radii/scale, matching ASE GUI's default atom sizing.
ROY_RADII = 0.5

DIMER_SCALE = 0.90
BULK_SCALE = 0.90
ROY_SCALE = 1.0

# Sized for direct placement on a 48 x 36 inch poster.
TITLE_FONT_SIZE = 30
LABEL_FONT_SIZE = 24
MISSING_FONT_SIZE = 20


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


def find_structure_by_name(root, structure_name, suffixes):
    """
    Find a structure file whose filename contains the requested name.
    """

    if not root.is_dir():
        return None

    suffixes = {suffix.lower() for suffix in suffixes}
    name_lower = structure_name.lower()

    matches = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in suffixes
            and name_lower in path.stem.lower()
        )
    ]

    if not matches:
        return None

    def match_rank(path):
        file_stem = path.stem.lower()

        if file_stem == name_lower:
            name_rank = 0
        elif file_stem.startswith(name_lower):
            name_rank = 1
        else:
            name_rank = 2

        return (
            name_rank,
            len(path.relative_to(root).parts),
            len(file_stem),
            str(path).lower(),
        )

    return min(matches, key=match_rank)


XYZ_ONLY_DIMER_NAMES = {
    "AlH3PH3",
    "BH3PH3",
    "BH3NH3",
    "CH3SiH3",
    "C2H6",
    "Si2H6",
}


def find_dimer_structure(system_name):
    """
    Locate the requested molecular or ionic-dimer geometry.
    """

    xyz_file = find_structure_by_name(
        root=XYZ_ROOT,
        structure_name=system_name,
        suffixes={".xyz"},
    )

    if xyz_file is not None:
        return xyz_file

    if system_name in XYZ_ONLY_DIMER_NAMES:
        print(
            f"[WARN] No matching .xyz file found for "
            f"{system_name} beneath {XYZ_ROOT}"
        )
        return None

    system_dir = DIMER_DIR / system_name

    candidates = [
        system_dir / "geometry.in",
        system_dir / "scale_1.000" / "geometry.in",
        system_dir / "reference" / "geometry.in",
        system_dir / "equilibrium" / "geometry.in",
        system_dir / "refined" / "geometry.in",
        system_dir / "refinement_100" / "geometry.in",
        DIMER_DIR / f"{system_name}_geometry.in",
        DIMER_DIR / f"{system_name}.in",
    ]

    structure_file = first_existing_path(candidates)

    if structure_file is not None:
        return structure_file

    if system_dir.is_dir():
        geometry_files = sorted(system_dir.rglob("geometry.in"))

        if geometry_files:
            preferred_terms = [
                "scale_1.000",
                "reference",
                "equilibrium",
                "refinement",
                "refined",
            ]

            for term in preferred_terms:
                for geometry_file in geometry_files:
                    if term in str(geometry_file):
                        return geometry_file

            return geometry_files[0]

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


def find_roy_structure(polymorph_name):
    """
    Locate one ROY polymorph from an EXTXYZ file beneath dft_calc.

    Matching requires an EXACT (case-insensitive) filename stem so that
    short, ambiguous names such as "Y" cannot accidentally match files
    like "YT04.extxyz" or "YN.extxyz".
    """

    if not EXTXYZ_ROOT.is_dir():
        return None

    name_lower = polymorph_name.lower()

    matches = [
        path
        for path in EXTXYZ_ROOT.rglob("*.extxyz")
        if path.stem.lower() == name_lower
    ]

    if not matches:
        print(
            f"[WARN] No exact .extxyz match for polymorph "
            f"'{polymorph_name}' beneath {EXTXYZ_ROOT}"
        )
        return None

    # Prefer the shallowest path if there happen to be multiple exact
    # matches in different subdirectories.
    matches.sort(key=lambda path: (len(path.relative_to(EXTXYZ_ROOT).parts), str(path).lower()))

    return matches[0]


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


def prepare_dimer_for_plotting(atoms, vacuum=7.0):
    """
    Place a nonperiodic dimer or molecule inside a cubic plotting cell.
    """

    atoms = atoms.copy()
    atoms.set_pbc(False)

    if len(atoms) == 0:
        return atoms

    positions = atoms.get_positions()

    extent = positions.max(axis=0) - positions.min(axis=0)
    largest_extent = max(extent)

    cell_length = max(
        largest_extent + 2.0 * vacuum,
        12.0,
    )

    atoms.set_cell(
        [
            [cell_length, 0.0, 0.0],
            [0.0, cell_length, 0.0],
            [0.0, 0.0, cell_length],
        ]
    )

    atoms.center()
    atoms.set_pbc(False)

    return atoms


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
        pad=22,
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
            pad=22,
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


def load_and_plot(axis, title, label, system_name, kind):
    """
    Locate, read, and plot one structure of the requested kind.
    """

    if kind == "dimer":
        structure_file = find_dimer_structure(system_name)
        atoms = read_structure(structure_file)

        if atoms is not None:
            atoms = prepare_dimer_for_plotting(atoms)

        rotation, radii, scale = DIMER_ROTATION, DIMER_RADII, DIMER_SCALE

    elif kind == "bulk":
        structure_file = find_bulk_structure(system_name)
        atoms = read_structure(structure_file)

        if atoms is not None:
            atoms = prepare_periodic_structure(atoms)

        rotation, radii, scale = BULK_ROTATION, BULK_RADII, BULK_SCALE

    else:  # roy
        structure_file = find_roy_structure(system_name)
        atoms = read_structure(structure_file)

        # Do not wrap: ASE GUI displays atoms at their original written
        # positions, and wrapping can shift molecules relative to the
        # cell in a way that no longer matches the GUI's default view.
        if atoms is not None:
            atoms = atoms.copy()
            atoms.set_pbc(True)

        rotation, radii, scale = ROY_ROTATION, ROY_RADII, ROY_SCALE

    plot_structure(
        axis=axis,
        atoms=atoms,
        title=title,
        label=label,
        rotation=rotation,
        radii=radii,
        scale=scale,
    )

    print(f"{kind:>6} {system_name:>10}: {structure_file}")


# =============================================================================
# Main routine
# =============================================================================

def main():
    print(f"Running script: {Path(__file__).resolve()}")
    print()

    figure = plt.figure(figsize=(24, 15))

    # 2 rows x 4 columns. Column 3 (the ROY panel) spans both rows.
    grid = gridspec.GridSpec(
        2,
        4,
        figure=figure,
        width_ratios=[1.0, 1.0, 1.0, 1.15],
        wspace=0.35,
        hspace=0.55,
    )

    # Top row: dimers.
    for column, (title, label, system_name) in enumerate(
        DIMER_STRUCTURES
    ):
        axis = figure.add_subplot(grid[0, column])

        load_and_plot(
            axis=axis,
            title=title,
            label=label,
            system_name=system_name,
            kind="dimer",
        )

    # Bottom row: bulk solids.
    for column, (title, label, system_name) in enumerate(
        BULK_STRUCTURES
    ):
        axis = figure.add_subplot(grid[1, column])

        load_and_plot(
            axis=axis,
            title=title,
            label=label,
            system_name=system_name,
            kind="bulk",
        )

    # Right column: ROY polymorph, spanning both rows.
    roy_title, roy_label, roy_system_name = ROY_STRUCTURE

    roy_axis = figure.add_subplot(grid[:, 3])

    load_and_plot(
        axis=roy_axis,
        title=roy_title,
        label=roy_label,
        system_name=roy_system_name,
        kind="roy",
    )

    figure.tight_layout(pad=3.0, w_pad=2.5, h_pad=3.0)

    figure.savefig(
        OUTPUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    print()
    print(f"Figure written to: {OUTPUT_FILE.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()


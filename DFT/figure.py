#!/usr/bin/env python3
"""
Create a three-column structural figure containing:

Column 1: Dimers
    - III-V semiconductors
    - Diamond-type covalent solids
    - Rock-salt ionic solids

Column 2: Bulk solids
    - III-V semiconductors
    - Diamond-type covalent solids
    - Rock-salt ionic solids

Column 3: ROY polymorphs
    - Y
    - YT04
    - R
    - OP
    - YN
    - ON
    - ORP

Each subtitle occupies its own grid row so that it does not overlap
the main title of the column.

Required packages:
    pip install ase matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch

from ase.io import read
from ase.visualize.plot import plot_atoms


# =============================================================================
# Configuration
# =============================================================================

DIMER_DIR = Path(
    "/mnt/ceph/users/cwoodson/dft_calc/diatomics/dimer_jobs"
)

BULK_DIR = Path(
    "/mnt/ceph/users/cwoodson/dft_calc/latbulk/eos_jobs"
)

ROY_DIR = Path(
    "/mnt/ceph/users/cwoodson/dft_calc/geometry_relax"
)

OUTPUT_FILE = Path("dimers_bulk_and_roy_structures.png")

DPI = 300


# =============================================================================
# Structure groups
# =============================================================================

DIMER_GROUPS = {
    "III–V semiconductors": [
        ("AlP dimer", "AlP"),
        ("BP dimer", "BP"),
        ("BN dimer", "BN"),
    ],
    "Diamond-type covalent solids": [
        ("SiC dimer", "SiC"),
        ("Ethane", "C2H6"),
        ("Disilane", "Si2H6"),
    ],
    "Rock-salt ionic solids": [
        ("LiCl dimer", "LiCl"),
        ("LiF dimer", "LiF"),
        ("NaCl dimer", "NaCl"),
        ("NaF dimer", "NaF"),
        ("MgO dimer", "MgO"),
        ("MgS dimer", "MgS"),
    ],
}

BULK_GROUPS = {
    "III–V semiconductors": [
        ("AlP", "AlP"),
        ("BP", "BP"),
        ("BN", "BN"),
    ],
    "Diamond-type covalent solids": [
        ("SiC", "SiC"),
        ("C", "C"),
        ("Si", "Si"),
    ],
    "Rock-salt ionic solids": [
        ("LiCl", "LiCl"),
        ("LiF", "LiF"),
        ("NaCl", "NaCl"),
        ("NaF", "NaF"),
        ("MgO", "MgO"),
        ("MgS", "MgS"),
    ],
}

ROY_POLYMORPHS = [
    ("Y", "Y"),
    ("YT04", "YT04"),
    ("R", "R"),
    ("OP", "OP"),
    ("YN", "YN"),
    ("ON", "ON"),
    ("ORP", "ORP"),
]


# =============================================================================
# Plot settings
# =============================================================================

DIMER_ROTATION = "10x,20y,0z"
BULK_ROTATION = "10x,20y,0z"
ROY_ROTATION = "15x,20y,0z"

DIMER_RADII = 0.75
BULK_RADII = 0.78
ROY_RADII = 0.42

DIMER_SCALE = 0.90
BULK_SCALE = 0.90
ROY_SCALE = 0.82

STRUCTURE_TITLE_FONT_SIZE = 9
GROUP_FONT_SIZE = 11
SECTION_FONT_SIZE = 15


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


def find_dimer_structure(system_name):
    """
    Locate a representative geometry for a dimer or molecule.
    """

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
        DIMER_DIR / f"{system_name}.xyz",
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
    Locate one ROY polymorph structure.
    """

    polymorph_dir = ROY_DIR / polymorph_name

    candidates = [
        ROY_DIR / f"{polymorph_name}.cif",
        ROY_DIR / f"{polymorph_name}.xyz",
        ROY_DIR / f"{polymorph_name}.in",
        ROY_DIR / f"{polymorph_name}_geometry.in",
        polymorph_dir / "geometry.in",
        polymorph_dir / "geometry.in.next_step",
        polymorph_dir / "optimized_geometry.in",
        polymorph_dir / "final_geometry.in",
        polymorph_dir / "aims.out",
    ]

    structure_file = first_existing_path(candidates)

    if structure_file is not None:
        return structure_file

    if polymorph_dir.is_dir():
        next_step_files = sorted(
            polymorph_dir.glob("geometry.in.next_step*")
        )

        if next_step_files:
            return next_step_files[-1]

        search_patterns = [
            "*.cif",
            "geometry.in",
            "*geometry*.in",
            "*.xyz",
            "aims.out",
        ]

        for pattern in search_patterns:
            matching_files = sorted(polymorph_dir.glob(pattern))

            if matching_files:
                return matching_files[-1]

    return None


# =============================================================================
# Structure reading and preparation
# =============================================================================

def read_structure(structure_file):
    """
    Read an ASE-supported structure file.

    For an FHI-aims output file, the final available structure is read.
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

    This cell is only used for visualization.
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


def show_missing_structure(axis, label):
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
        fontsize=8,
        color="crimson",
    )

    axis.set_title(
        label,
        fontsize=STRUCTURE_TITLE_FONT_SIZE,
        pad=3,
    )


def plot_structure(
    axis,
    atoms,
    label,
    rotation,
    radii,
    scale,
    show_cell,
):
    """
    Plot one ASE Atoms object.
    """

    clear_axis(axis)

    if atoms is None:
        show_missing_structure(axis, label)
        return

    try:
        plot_atoms(
            atoms,
            axis,
            rotation=rotation,
            radii=radii,
            scale=scale,
            show_unit_cell=show_cell,
        )

        axis.set_title(
            label,
            fontsize=STRUCTURE_TITLE_FONT_SIZE,
            pad=3,
        )

        axis.set_aspect("equal")
        clear_axis(axis)

    except Exception as error:
        print(f"[WARN] Could not plot {label}: {error}")
        show_missing_structure(axis, label)


def format_subtitle_axis(axis, title):
    """
    Format a dedicated grid row as a gray category subtitle.

    Using a real grid row prevents the subtitle from overlapping
    the main panel title.
    """

    axis.set_facecolor("0.96")

    axis.text(
        0.5,
        0.5,
        title,
        ha="center",
        va="center",
        transform=axis.transAxes,
        fontsize=GROUP_FONT_SIZE,
        fontweight="bold",
    )

    axis.set_xticks([])
    axis.set_yticks([])

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("0.65")


def add_panel_border(
    figure,
    structure_axes,
    subtitle_axes,
    title,
    title_height=0.026,
    title_gap=0.007,
    padding=0.006,
):
    """
    Add a rounded border and main title around a complete column.

    The subtitle axes are included when calculating the panel bounds.
    Extra vertical space is reserved above the highest subtitle for
    the main panel title.
    """

    all_axes = list(structure_axes) + list(subtitle_axes)
    positions = [axis.get_position() for axis in all_axes]

    left = min(position.x0 for position in positions) - padding
    right = max(position.x1 for position in positions) + padding
    bottom = min(position.y0 for position in positions) - padding

    content_top = max(position.y1 for position in positions)

    title_bottom = content_top + title_gap
    top = title_bottom + title_height + padding

    border = FancyBboxPatch(
        (left, bottom),
        right - left,
        top - bottom,
        boxstyle="round,pad=0.004",
        linewidth=1.2,
        edgecolor="black",
        facecolor="none",
        transform=figure.transFigure,
        clip_on=False,
        zorder=10,
    )

    figure.add_artist(border)

    figure.text(
        0.5 * (left + right),
        title_bottom + 0.5 * title_height,
        title,
        ha="center",
        va="center",
        fontsize=SECTION_FONT_SIZE,
        fontweight="bold",
        zorder=11,
    )


# =============================================================================
# Dimer column
# =============================================================================

def create_dimer_panel(figure, parent_spec):
    """
    Create the first column containing the dimer structures.
    """

    panel_grid = gridspec.GridSpecFromSubplotSpec(
        6,
        3,
        subplot_spec=parent_spec,
        height_ratios=[
            0.16,
            1.00,
            0.16,
            1.00,
            0.16,
            2.00,
        ],
        hspace=0.14,
        wspace=0.08,
    )

    all_axes = []
    subtitle_axes = []

    # -------------------------------------------------------------------------
    # III-V semiconductors
    # -------------------------------------------------------------------------

    subtitle_axis = figure.add_subplot(
        panel_grid[0, :]
    )

    format_subtitle_axis(
        subtitle_axis,
        "III–V semiconductors",
    )

    subtitle_axes.append(subtitle_axis)

    group_name = "III–V semiconductors"

    for column, (label, system_name) in enumerate(
        DIMER_GROUPS[group_name]
    ):
        axis = figure.add_subplot(
            panel_grid[1, column]
        )

        structure_file = find_dimer_structure(system_name)
        atoms = read_structure(structure_file)

        if atoms is not None:
            atoms = prepare_dimer_for_plotting(atoms)

        plot_structure(
            axis=axis,
            atoms=atoms,
            label=label,
            rotation=DIMER_ROTATION,
            radii=DIMER_RADII,
            scale=DIMER_SCALE,
            show_cell=2,
        )

        all_axes.append(axis)

        print(f"Dimer {system_name:>6}: {structure_file}")

    # -------------------------------------------------------------------------
    # Diamond-type covalent solids
    # -------------------------------------------------------------------------

    subtitle_axis = figure.add_subplot(
        panel_grid[2, :]
    )

    format_subtitle_axis(
        subtitle_axis,
        "Diamond-type covalent solids",
    )

    subtitle_axes.append(subtitle_axis)

    group_name = "Diamond-type covalent solids"

    for column, (label, system_name) in enumerate(
        DIMER_GROUPS[group_name]
    ):
        axis = figure.add_subplot(
            panel_grid[3, column]
        )

        structure_file = find_dimer_structure(system_name)
        atoms = read_structure(structure_file)

        if atoms is not None:
            atoms = prepare_dimer_for_plotting(atoms)

        plot_structure(
            axis=axis,
            atoms=atoms,
            label=label,
            rotation=DIMER_ROTATION,
            radii=DIMER_RADII,
            scale=DIMER_SCALE,
            show_cell=2,
        )

        all_axes.append(axis)

        print(f"Dimer {system_name:>6}: {structure_file}")

    # -------------------------------------------------------------------------
    # Rock-salt ionic solids
    # -------------------------------------------------------------------------

    subtitle_axis = figure.add_subplot(
        panel_grid[4, :]
    )

    format_subtitle_axis(
        subtitle_axis,
        "Rock-salt ionic solids",
    )

    subtitle_axes.append(subtitle_axis)

    group_name = "Rock-salt ionic solids"

    rock_salt_grid = gridspec.GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=panel_grid[5, :],
        hspace=0.16,
        wspace=0.08,
    )

    for index, (label, system_name) in enumerate(
        DIMER_GROUPS[group_name]
    ):
        row = index // 3
        column = index % 3

        axis = figure.add_subplot(
            rock_salt_grid[row, column]
        )

        structure_file = find_dimer_structure(system_name)
        atoms = read_structure(structure_file)

        if atoms is not None:
            atoms = prepare_dimer_for_plotting(atoms)

        plot_structure(
            axis=axis,
            atoms=atoms,
            label=label,
            rotation=DIMER_ROTATION,
            radii=DIMER_RADII,
            scale=DIMER_SCALE,
            show_cell=2,
        )

        all_axes.append(axis)

        print(f"Dimer {system_name:>6}: {structure_file}")

    return all_axes, subtitle_axes


# =============================================================================
# Bulk-solid column
# =============================================================================

def create_bulk_panel(figure, parent_spec):
    """
    Create the second column containing the bulk-solid structures.
    """

    panel_grid = gridspec.GridSpecFromSubplotSpec(
        6,
        3,
        subplot_spec=parent_spec,
        height_ratios=[
            0.16,
            1.00,
            0.16,
            1.00,
            0.16,
            2.00,
        ],
        hspace=0.14,
        wspace=0.08,
    )

    all_axes = []
    subtitle_axes = []

    # -------------------------------------------------------------------------
    # III-V semiconductors
    # -------------------------------------------------------------------------

    subtitle_axis = figure.add_subplot(
        panel_grid[0, :]
    )

    format_subtitle_axis(
        subtitle_axis,
        "III–V semiconductors",
    )

    subtitle_axes.append(subtitle_axis)

    group_name = "III–V semiconductors"

    for column, (label, system_name) in enumerate(
        BULK_GROUPS[group_name]
    ):
        axis = figure.add_subplot(
            panel_grid[1, column]
        )

        structure_file = find_bulk_structure(system_name)
        atoms = read_structure(structure_file)

        if atoms is not None:
            atoms = prepare_periodic_structure(atoms)

        plot_structure(
            axis=axis,
            atoms=atoms,
            label=label,
            rotation=BULK_ROTATION,
            radii=BULK_RADII,
            scale=BULK_SCALE,
            show_cell=2,
        )

        all_axes.append(axis)

        print(f"Bulk  {system_name:>6}: {structure_file}")

    # -------------------------------------------------------------------------
    # Diamond-type covalent solids
    # -------------------------------------------------------------------------

    subtitle_axis = figure.add_subplot(
        panel_grid[2, :]
    )

    format_subtitle_axis(
        subtitle_axis,
        "Diamond-type covalent solids",
    )

    subtitle_axes.append(subtitle_axis)

    group_name = "Diamond-type covalent solids"

    for column, (label, system_name) in enumerate(
        BULK_GROUPS[group_name]
    ):
        axis = figure.add_subplot(
            panel_grid[3, column]
        )

        structure_file = find_bulk_structure(system_name)
        atoms = read_structure(structure_file)

        if atoms is not None:
            atoms = prepare_periodic_structure(atoms)

        plot_structure(
            axis=axis,
            atoms=atoms,
            label=label,
            rotation=BULK_ROTATION,
            radii=BULK_RADII,
            scale=BULK_SCALE,
            show_cell=2,
        )

        all_axes.append(axis)

        print(f"Bulk  {system_name:>6}: {structure_file}")

    # -------------------------------------------------------------------------
    # Rock-salt ionic solids
    # -------------------------------------------------------------------------

    subtitle_axis = figure.add_subplot(
        panel_grid[4, :]
    )

    format_subtitle_axis(
        subtitle_axis,
        "Rock-salt ionic solids",
    )

    subtitle_axes.append(subtitle_axis)

    group_name = "Rock-salt ionic solids"

    rock_salt_grid = gridspec.GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=panel_grid[5, :],
        hspace=0.16,
        wspace=0.08,
    )

    for index, (label, system_name) in enumerate(
        BULK_GROUPS[group_name]
    ):
        row = index // 3
        column = index % 3

        axis = figure.add_subplot(
            rock_salt_grid[row, column]
        )

        structure_file = find_bulk_structure(system_name)
        atoms = read_structure(structure_file)

        if atoms is not None:
            atoms = prepare_periodic_structure(atoms)

        plot_structure(
            axis=axis,
            atoms=atoms,
            label=label,
            rotation=BULK_ROTATION,
            radii=BULK_RADII,
            scale=BULK_SCALE,
            show_cell=2,
        )

        all_axes.append(axis)

        print(f"Bulk  {system_name:>6}: {structure_file}")

    return all_axes, subtitle_axes


# =============================================================================
# ROY column
# =============================================================================

def create_roy_panel(figure, parent_spec):
    """
    Create the third column containing the seven ROY polymorphs.

    Layout:

        Y       YT04
        R       OP
        YN      ON
           ORP
    """

    panel_grid = gridspec.GridSpecFromSubplotSpec(
        5,
        2,
        subplot_spec=parent_spec,
        height_ratios=[
            0.16,
            1.00,
            1.00,
            1.00,
            1.00,
        ],
        hspace=0.14,
        wspace=0.08,
    )

    subtitle_axis = figure.add_subplot(
        panel_grid[0, :]
    )

    format_subtitle_axis(
        subtitle_axis,
        "ROY polymorphs",
    )

    subtitle_axes = [subtitle_axis]
    all_axes = []

    positions = [
        (1, 0),              # Y
        (1, 1),              # YT04
        (2, 0),              # R
        (2, 1),              # OP
        (3, 0),              # YN
        (3, 1),              # ON
        (4, slice(0, 2)),    # ORP
    ]

    for (label, polymorph_name), (row, column) in zip(
        ROY_POLYMORPHS,
        positions,
    ):
        axis = figure.add_subplot(
            panel_grid[row, column]
        )

        structure_file = find_roy_structure(polymorph_name)
        atoms = read_structure(structure_file)

        if atoms is not None:
            atoms = prepare_periodic_structure(atoms)

        plot_structure(
            axis=axis,
            atoms=atoms,
            label=label,
            rotation=ROY_ROTATION,
            radii=ROY_RADII,
            scale=ROY_SCALE,
            show_cell=2,
        )

        all_axes.append(axis)

        print(f"ROY   {polymorph_name:>6}: {structure_file}")

    return all_axes, subtitle_axes


# =============================================================================
# Main routine
# =============================================================================

def main():
    figure = plt.figure(
        figsize=(24, 14),
        constrained_layout=False,
    )

    outer_grid = gridspec.GridSpec(
        1,
        3,
        figure=figure,
        width_ratios=[
            1.0,
            1.0,
            0.72,
        ],
        left=0.02,
        right=0.98,
        bottom=0.03,
        top=0.925,
        wspace=0.055,
    )

    # First column: dimers.
    dimer_axes, dimer_subtitle_axes = create_dimer_panel(
        figure,
        outer_grid[0, 0],
    )

    # Second column: bulk solids.
    bulk_axes, bulk_subtitle_axes = create_bulk_panel(
        figure,
        outer_grid[0, 1],
    )

    # Third column: ROY polymorphs.
    roy_axes, roy_subtitle_axes = create_roy_panel(
        figure,
        outer_grid[0, 2],
    )

    # Finalize all axes before adding the panel borders and main titles.
    figure.canvas.draw()

    add_panel_border(
        figure=figure,
        structure_axes=dimer_axes,
        subtitle_axes=dimer_subtitle_axes,
        title="Dimers",
    )

    add_panel_border(
        figure=figure,
        structure_axes=bulk_axes,
        subtitle_axes=bulk_subtitle_axes,
        title="Bulk solids",
    )

    add_panel_border(
        figure=figure,
        structure_axes=roy_axes,
        subtitle_axes=roy_subtitle_axes,
        title="Molecular crystals",
    )

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

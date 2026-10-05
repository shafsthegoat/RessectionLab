"""Native case inspection and route comparison workspace.

Imaging, planning and persistence live outside the UI. Importing this package
does not initialize Qt, VTK, or a graphical session.
"""


def main() -> int:
    """Launch the local desktop application."""
    from .desktop import main as launch

    return launch()

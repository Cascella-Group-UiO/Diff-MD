from pathlib import Path

import numpy as np
import pytest

from gmx2HyMD.gro_utils import GroAtom, load_gro


def _gro_atom_line(resid=1, resname="SOL", atom="OW", index=1, x=0.1, y=0.2, z=0.3) -> str:
    return f"{resid:5d}{resname:<5}{atom:>5}{index:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"


def test_parse_line_accepts_position_only_gro_line() -> None:
    atom = GroAtom.parse_line(_gro_atom_line())

    assert atom.resid == 1
    assert atom.resname == "SOL"
    assert atom.atom_name == "OW"
    assert atom.index == 1
    assert (atom.x, atom.y, atom.z) == pytest.approx((0.1, 0.2, 0.3))
    assert (atom.vx, atom.vy, atom.vz) == (0.0, 0.0, 0.0)


def test_parse_line_reports_actual_bad_line_length() -> None:
    with pytest.raises(ValueError, match="line length is 5"):
        GroAtom.parse_line("short")


def test_load_gro_rejects_incomplete_file(tmp_path: Path) -> None:
    gro_path = tmp_path / "empty.gro"
    gro_path.write_text("title\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        load_gro(str(gro_path))


def test_load_gro_rejects_malformed_box(tmp_path: Path) -> None:
    gro_path = tmp_path / "bad_box.gro"
    gro_path.write_text(
        "title\n"
        "1\n"
        f"{_gro_atom_line()}"
        "1.0 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported box line"):
        load_gro(str(gro_path))


def test_load_gro_reads_declared_atoms_and_box(tmp_path: Path) -> None:
    gro_path = tmp_path / "ok.gro"
    gro_path.write_text(
        "title\n"
        "1\n"
        f"{_gro_atom_line()}"
        "1.0 2.0 3.0\n",
        encoding="utf-8",
    )

    atoms, box = load_gro(str(gro_path))

    assert len(atoms) == 1
    assert np.allclose(box, np.asarray([1.0, 2.0, 3.0]))

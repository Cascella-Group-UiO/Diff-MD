from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np
import pytest

from gmx2HyMD.cmap_utils import (
    CMAP_GRID_SIZE,
    build_cmap_entries_for_chain,
    cmap_grids_to_toml_str,
    parse_cmap_itp,
)


def _write_cmap(path: Path, header: str, values: list[float]) -> None:
    value_text = " ".join(str(value) for value in values)
    path.write_text(
        "[ cmaptypes ]\n"
        "; comments should be ignored\n"
        f"{header} \\\n"
        f"{value_text}\n",
        encoding="utf-8",
    )


def test_parse_cmap_itp_small_grid_and_toml_round_trip(tmp_path: Path) -> None:
    cmap_path = tmp_path / "cmap.itp"
    _write_cmap(cmap_path, "C-* N-ALA XC-ALA C-ALA N-* 1 2 2", [0.0, 1.0, 2.0, 3.0])

    grids = parse_cmap_itp(str(cmap_path), grid_size=2)

    assert set(grids) == {"ALA"}
    assert np.allclose(grids["ALA"], np.asarray([[0.0, 1.0], [2.0, 3.0]]))

    parsed = tomllib.loads(cmap_grids_to_toml_str(grids, grid_size=2))
    assert parsed["cmap"]["grid_size"] == 2
    assert parsed["cmap"]["grids"]["ALA"] == [[0.0, 1.0], [2.0, 3.0]]


def test_parse_real_amber19sb_cmap_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parents[2]
        / "reference_examples"
        / "cmap_amber19"
        / "amber19sb.ff"
        / "cmap.itp"
    )
    if not fixture.exists():
        pytest.skip("amber19sb cmap.itp reference fixture not bundled")

    grids = parse_cmap_itp(str(fixture))

    assert "ALA" in grids
    assert "GLY" in grids
    assert all(grid.shape == (CMAP_GRID_SIZE, CMAP_GRID_SIZE) for grid in grids.values())


@pytest.mark.parametrize(
    ("header", "values", "match"),
    [
        ("C-* N-ALA XC-ALA C-ALA N-* 1 3 2", [0.0] * 6, "grid size mismatch"),
        ("C-* N-ALA XC-ALA C-ALA N-* 2 2 2", [0.0] * 4, "function type must be 1"),
        ("C-* N-ALA XC-ALA C-ALA N-* 1 2 2", [0.0] * 3, "Truncated CMAP grid"),
    ],
)
def test_parse_cmap_itp_rejects_malformed_entries(tmp_path: Path, header: str, values: list[float], match: str) -> None:
    cmap_path = tmp_path / "bad_cmap.itp"
    _write_cmap(cmap_path, header, values)

    with pytest.raises(ValueError, match=match):
        parse_cmap_itp(str(cmap_path), grid_size=2)


def test_parse_cmap_itp_rejects_partial_trailing_record(tmp_path: Path) -> None:
    cmap_path = tmp_path / "bad_cmap.itp"
    cmap_path.write_text(
        "[ cmaptypes ]\n"
        "C-* N-ALA XC-ALA C-ALA N-* 1 2 2 0 1 2 3\n"
        "C-* N-GLY\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Malformed CMAP entry"):
        parse_cmap_itp(str(cmap_path), grid_size=2)


@dataclass
class FakeAtom:
    index: int
    atomname: str
    resnr: int
    resname: str


def _residue(resnr: int, resname: str, offset: int) -> list[FakeAtom]:
    return [
        FakeAtom(offset + 1, "N", resnr, resname),
        FakeAtom(offset + 2, "CA", resnr, resname),
        FakeAtom(offset + 3, "C", resnr, resname),
    ]


def test_build_cmap_entries_for_chain() -> None:
    atoms = _residue(1, "GLY", 0) + _residue(2, "ALA", 3) + _residue(3, "GLY", 6)
    grids = {"ALA": np.zeros((2, 2), dtype=np.float64)}

    entries = build_cmap_entries_for_chain(atoms, grids)

    assert entries == [[3, 4, 5, 6, 7, "ALA"]]


def test_build_cmap_entries_skips_missing_grid(capsys) -> None:
    atoms = _residue(1, "GLY", 0) + _residue(2, "SER", 3) + _residue(3, "GLY", 6)

    entries = build_cmap_entries_for_chain(atoms, {"ALA": np.zeros((2, 2), dtype=np.float64)})

    assert entries == []
    assert "SER: 1" in capsys.readouterr().out

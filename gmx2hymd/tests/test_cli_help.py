import sys

import pytest

from gmx2HyMD import analyze_cli, main, traj_cli


def test_analyze_help_mentions_q5_site5_and_tau5() -> None:
    help_text = analyze_cli.build_parser().format_help()

    assert "--q5" in help_text
    assert "--site5" in help_text
    assert "tau5" in help_text


def test_traj_help_preserves_examples() -> None:
    help_text = traj_cli.build_parser().format_help()

    assert "Examples" in help_text
    assert "diffmd-h5toxyz -f simulation.h5" in help_text
    assert "--gro-to-h5" in help_text


def test_main_help_preserves_examples(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["gmx2diffmd", "-h"])

    with pytest.raises(SystemExit) as excinfo:
        main.user_input()

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "Examples" in help_text
    assert "--amber19sb" in help_text
    assert "cmap.itp" in help_text

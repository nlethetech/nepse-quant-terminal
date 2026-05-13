from pathlib import Path

from scripts.launch import bootstrap


def test_supported_python_range():
    assert bootstrap.is_supported_python((3, 10, 0))
    assert bootstrap.is_supported_python((3, 12, 4))
    assert bootstrap.is_supported_python((3, 13, 9))
    assert not bootstrap.is_supported_python((3, 9, 18))
    assert not bootstrap.is_supported_python((3, 14, 0))


def test_requirements_hash_changes_with_file_content(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("textual\n", encoding="utf-8")
    first = bootstrap.requirements_hash(path)
    path.write_text("textual\nrich\n", encoding="utf-8")
    second = bootstrap.requirements_hash(path)

    assert first != second


def test_venv_python_path_is_platform_specific(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap.os, "name", "nt")
    assert bootstrap.venv_python(tmp_path) == tmp_path / ".venv" / "Scripts" / "python.exe"

    monkeypatch.setattr(bootstrap.os, "name", "posix")
    assert bootstrap.venv_python(tmp_path) == tmp_path / ".venv" / "bin" / "python"

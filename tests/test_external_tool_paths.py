"""External programs run by an absolute path found on PATH (utils/tool_paths.py).

Covers the search itself (``find_tool``: absolute PATH entries only, in order,
PATHEXT on Windows, the execute bit elsewhere, never the current directory) and the
child environments (a filtered one keeps PATH, PATHEXT, SYSTEMROOT and
NoDefaultCurrentDirectoryInExePath).

The ``tools`` fixture reproduces the Windows default by removing
NoDefaultCurrentDirectoryInExePath from the test process (some shells export it).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from soup_cli.utils.tool_paths import (
    ALWAYS_KEPT_ENV_NAMES,
    ToolNotFoundError,
    filtered_environment,
    find_tool,
    require_tool,
    tool_environment,
)
from tests.conftest import strip_ansi

_WINDOWS = os.name == "nt"
_PATHEXT = ".COM;.EXE;.BAT;.CMD"
windows_only = pytest.mark.skipif(not _WINDOWS, reason="PATHEXT names Windows program files")
posix_only = pytest.mark.skipif(_WINDOWS, reason="Windows has no execute permission bit")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(strip_ansi(text).split())


def _program(directory: Path, name: str, *, suffix: str | None = None) -> Path:
    """An executable file named as this OS names programs (``.EXE`` on Windows)."""
    directory.mkdir(parents=True, exist_ok=True)
    if suffix is None:
        suffix = ".EXE" if _WINDOWS else ""
    path = directory / (name + suffix)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _same(left: str, right: object) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """``install(name)`` puts one copy of a program in the current directory, one
    under the relative PATH entry ``rel`` and one in an absolute PATH directory,
    and returns the absolute path of that last copy: the only one to be run."""
    project = tmp_path / "project"
    installed = tmp_path / "installed"
    project.mkdir()
    installed.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", "rel", str(installed)]))
    monkeypatch.setenv("PATHEXT", _PATHEXT)
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)

    def install(name: str) -> str:
        _program(project, name)
        _program(project / "rel", name)
        return os.path.abspath(_program(installed, name))

    install.project = project
    install.installed = installed
    return install


@pytest.fixture
def no_tools(tmp_path, monkeypatch):
    """A PATH with one empty absolute directory: no program is found."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setenv("PATHEXT", _PATHEXT)
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)


class _Recorder:
    """A stand-in for ``subprocess.run`` that records each argv and env."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr

    def __call__(self, argv, *args, **kwargs):
        self.calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, self.stderr)

    @property
    def argv0(self) -> list[str]:
        return [argv[0] for argv, _ in self.calls]


# ---------------------------------------------------------------------------
# find_tool: the search
# ---------------------------------------------------------------------------


class TestTheSearch:
    def test_the_copy_in_the_absolute_path_directory_is_the_one_found(self, tools):
        expected = tools("soup-probe-tool")

        assert _same(find_tool("soup-probe-tool"), expected)

    def test_a_program_only_in_the_current_or_a_relative_directory_is_not_found(
        self, tools
    ):
        _program(tools.project, "soup-probe-tool")
        _program(tools.project / "rel", "soup-probe-tool")

        assert find_tool("soup-probe-tool") is None

    def test_empty_dot_and_relative_entries_are_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for directory in (tmp_path, tmp_path / "bin", tmp_path / "rel" / "bin"):
            _program(directory, "soup-probe-tool")
        relative = ["", " ", ".", "./bin", "bin", os.path.join("rel", "bin"), '""']
        monkeypatch.setenv("PATH", os.pathsep.join(relative))
        monkeypatch.setenv("PATHEXT", _PATHEXT)
        monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)

        assert find_tool("soup-probe-tool") is None

    def test_the_entries_are_searched_in_order(self, tmp_path, monkeypatch):
        first = _program(tmp_path / "first", "soup-probe-tool")
        _program(tmp_path / "second", "soup-probe-tool")
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(tmp_path / "first"), str(tmp_path / "second")])
        )
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert _same(find_tool("soup-probe-tool"), first)

    def test_a_quoted_entry_with_spaces_is_searched(self, tmp_path, monkeypatch):
        program = _program(tmp_path / "a dir with spaces", "soup-probe-tool")
        monkeypatch.setenv("PATH", f' "{tmp_path / "a dir with spaces"}" ')
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert _same(find_tool("soup-probe-tool"), program)

    def test_the_result_is_absolute_and_normalised(self, tmp_path, monkeypatch):
        program = _program(tmp_path / "bin", "soup-probe-tool")
        monkeypatch.setenv("PATH", str(tmp_path / "bin" / ".." / "bin"))
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        found = find_tool("soup-probe-tool")

        assert os.path.isabs(found)
        assert ".." not in Path(found).parts
        assert _same(found, program)

    def test_a_directory_named_like_the_program_is_skipped(self, tmp_path, monkeypatch):
        decoy = tmp_path / "first" / ("soup-probe-tool" + (".EXE" if _WINDOWS else ""))
        decoy.mkdir(parents=True)
        program = _program(tmp_path / "second", "soup-probe-tool")
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(tmp_path / "first"), str(tmp_path / "second")])
        )
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert _same(find_tool("soup-probe-tool"), program)

    def test_an_explicit_search_path_replaces_the_path_variable(self, tmp_path, monkeypatch):
        program = _program(tmp_path / "elsewhere", "soup-probe-tool")
        monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert find_tool("soup-probe-tool") is None
        assert _same(find_tool("soup-probe-tool", path=str(tmp_path / "elsewhere")), program)
        assert find_tool("soup-probe-tool", path="") is None

    def test_an_unset_path_does_not_search_the_current_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _program(tmp_path, "soup-probe-tool")
        monkeypatch.delenv("PATH", raising=False)
        monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)

        assert find_tool("soup-probe-tool") is None

    @pytest.mark.parametrize(
        "name", ["", ".", "..", "bin/soup-probe-tool", os.path.join("bin", "soup-probe-tool")]
    )
    def test_a_name_with_a_directory_part_is_refused(self, name):
        with pytest.raises(ValueError, match="bare program name"):
            find_tool(name)

    @posix_only
    def test_a_file_without_the_execute_bit_is_skipped(self, tmp_path, monkeypatch):
        plain = tmp_path / "first" / "soup-probe-tool"
        plain.parent.mkdir()
        plain.write_text("data\n", encoding="utf-8")
        plain.chmod(0o644)
        program = _program(tmp_path / "second", "soup-probe-tool")
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(tmp_path / "first"), str(tmp_path / "second")])
        )

        assert _same(find_tool("soup-probe-tool"), program)

    @windows_only
    def test_each_pathext_extension_is_tried_in_its_order(self, tmp_path, monkeypatch):
        directory = tmp_path / "bin"
        cmd = _program(directory, "soup-probe-tool", suffix=".CMD")
        monkeypatch.setenv("PATH", str(directory))
        monkeypatch.setenv("PATHEXT", ".EXE;.CMD")

        assert _same(find_tool("soup-probe-tool"), cmd)

        exe = _program(directory, "soup-probe-tool", suffix=".EXE")
        assert _same(find_tool("soup-probe-tool"), exe)

    @windows_only
    def test_the_default_extensions_apply_when_pathext_is_unset(self, tmp_path, monkeypatch):
        bat = _program(tmp_path / "bin", "soup-probe-tool", suffix=".BAT")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.delenv("PATHEXT", raising=False)

        assert _same(find_tool("soup-probe-tool"), bat)

    @windows_only
    def test_a_name_that_ends_in_an_extension_is_tried_as_it_is(self, tmp_path, monkeypatch):
        cmd = _program(tmp_path / "bin", "soup-probe-tool", suffix=".cmd")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.setenv("PATHEXT", ".EXE;.CMD")

        assert _same(find_tool("soup-probe-tool.cmd"), cmd)
        assert find_tool("soup-probe-tool.bat") is None

    @windows_only
    def test_a_file_without_an_extension_is_not_a_program(self, tmp_path, monkeypatch):
        _program(tmp_path / "bin", "soup-probe-tool", suffix="")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert find_tool("soup-probe-tool") is None

    @windows_only
    def test_rooted_and_drive_relative_entries_are_skipped(self, tmp_path, monkeypatch):
        _program(tmp_path / "bin", "soup-probe-tool")
        drive, rooted = os.path.splitdrive(str(tmp_path / "bin"))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", os.pathsep.join([rooted, drive + "bin"]))
        monkeypatch.setenv("PATHEXT", _PATHEXT)

        assert find_tool("soup-probe-tool") is None


class TestRequireTool:
    def test_the_absolute_path_is_returned(self, tools):
        expected = tools("soup-probe-tool")

        assert _same(require_tool("soup-probe-tool", "Install it."), expected)

    def test_a_missing_program_is_named_with_how_to_install_it(self, no_tools):
        with pytest.raises(ToolNotFoundError) as excinfo:
            require_tool("soup-probe-tool", "Install it from https://example.invalid.")

        assert isinstance(excinfo.value, FileNotFoundError)
        message = str(excinfo.value)
        assert "soup-probe-tool was not found on PATH" in message
        assert "Install it from https://example.invalid." in message


# ---------------------------------------------------------------------------
# Child environments
# ---------------------------------------------------------------------------

_FOUR = {"PATH", "PATHEXT", "SYSTEMROOT", "NoDefaultCurrentDirectoryInExePath"}


def _folded(env) -> set[str]:
    return {name.upper() for name in env}


class TestFilteredEnvironment:
    def test_the_four_program_search_names_are_always_kept(self):
        assert set(ALWAYS_KEPT_ENV_NAMES) == _FOUR

    def test_exact_matching_keeps_the_named_and_the_four(self):
        environ = {
            "PATH": "/usr/bin",
            "PATHEXT": ".EXE",
            "SYSTEMROOT": "C:\\Windows",
            "NoDefaultCurrentDirectoryInExePath": "1",
            "LANG": "C.UTF-8",
            "HF_TOKEN": "hf-secret",
            "lang": "lower-case is another variable here",
        }

        env = filtered_environment({"LANG"}, environ, case_insensitive=False)

        assert env == {
            "PATH": "/usr/bin",
            "PATHEXT": ".EXE",
            "SYSTEMROOT": "C:\\Windows",
            "NoDefaultCurrentDirectoryInExePath": "1",
            "LANG": "C.UTF-8",
        }

    def test_windows_matching_ignores_case(self):
        environ = {
            "Path": "C:\\bin",
            "PathExt": ".EXE",
            "SystemRoot": "C:\\Windows",
            "NODEFAULTCURRENTDIRECTORYINEXEPATH": "1",
            "OPENAI_API_KEY": "sk-secret",
        }

        env = filtered_environment((), environ, case_insensitive=True)

        assert _folded(env) == {name.upper() for name in _FOUR}
        assert "OPENAI_API_KEY" not in env

    def test_the_default_follows_the_platform(self, monkeypatch):
        monkeypatch.setenv("SOUP_PROBE_VARIABLE", "x")
        monkeypatch.setenv("PATH", os.environ["PATH"])

        env = filtered_environment(())

        assert "PATH" in _folded(env)
        assert "SOUP_PROBE_VARIABLE" not in _folded(env)


class TestToolEnvironment:
    def test_the_named_variables_are_left_out_ignoring_case(self):
        environ = {"RUNPOD_API_KEY": "k", "runpod_api_key": "k2", "PATH": "/bin", "HOME": "/h"}

        env = tool_environment(drop=("RunPod_Api_Key",), environ=environ)

        assert "RUNPOD_API_KEY" not in _folded(env)
        assert env["PATH"] == "/bin"
        assert env["HOME"] == "/h"

    def test_windows_programs_skip_the_current_directory_too(self, monkeypatch):
        monkeypatch.setenv("NoDefaultCurrentDirectoryInExePath", "0")

        env = tool_environment()

        names = [name for name in env if name.upper() == "NODEFAULTCURRENTDIRECTORYINEXEPATH"]
        if _WINDOWS:
            assert names == ["NoDefaultCurrentDirectoryInExePath"]
            assert env["NoDefaultCurrentDirectoryInExePath"] == "1"
        else:
            assert names == ["NoDefaultCurrentDirectoryInExePath"]
            assert env["NoDefaultCurrentDirectoryInExePath"] == "0"

    def test_the_result_is_a_new_mapping(self):
        assert tool_environment() is not os.environ


def _with_the_four(monkeypatch) -> None:
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setenv("PATHEXT", _PATHEXT)
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    monkeypatch.setenv("NoDefaultCurrentDirectoryInExePath", "1")

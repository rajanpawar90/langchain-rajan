from typing import Dict, Optional

import pytest

from langchain_cli.constants import (
    DEFAULT_GIT_REF,
    DEFAULT_GIT_REPO,
    DEFAULT_GIT_SUBDIRECTORY,
)
from langchain_cli.utils.git import DependencySource, parse_dependency_string


def _assert_dependency_equals(dep: DependencySource, expected: DependencySource) -> None:
    assert dep == expected


def test_dependency_string() -> None:
    test_cases = [
        (
            "git+ssh://git@github.com/efriis/myrepo.git",
            {"git": "ssh://git@github.com/efriis/myrepo.git", "ref": None, "subdirectory": None},
        ),
        (
            "git+https://github.com/efriis/myrepo.git#subdirectory=src",
            {"git": "https://github.com/efriis/myrepo.git", "subdirectory": "src", "ref": None},
        ),
        (
            "git+ssh://git@github.com:efriis/myrepo.git#develop",
            {"git": "ssh://git@github.com:efriis/myrepo.git", "ref": "develop", "subdirectory": None},
        ),
        (
            "git+ssh://git@github.com/efriis/myrepo.git#develop",
            {"git": "ssh://git@github.com/efriis/myrepo.git", "ref": "develop", "subdirectory": None},
        ),
        (
            "git+ssh://git@github.com:efriis/myrepo.git@develop",
            {"git": "ssh://git@github.com:efriis/myrepo.git", "ref": "develop", "subdirectory": None},
        ),
        (
            "simple-pirate",
            {
                "git": DEFAULT_GIT_REPO,
                "subdirectory": f"{DEFAULT_GIT_SUBDIRECTORY}/simple-pirate",
                "ref": DEFAULT_GIT_REF,
            },
        ),
    ]

    for input_string, expected_dep in test_cases:
        parsed_dep = parse_dependency_string(input_string, None, None, None)
        _assert_dependency_equals(parsed_dep, expected_dep)


def test_dependency_string_both() -> None:
    test_cases = [
        (
            "git+https://github.com/efriis/myrepo.git@branch#subdirectory=src",
            {
                "git": "https://github.com/efriis/myrepo.git",
                "subdirectory": "src",
                "ref": "branch",
            },
        ),
    ]

    for input_string, expected_dep in test_cases:
        parsed_dep = parse_dependency_string(input_string, None, None, None)
        _assert_dependency_equals(parsed_dep, expected_dep)


def test_dependency_string_invalids() -> None:
    with pytest.raises(ValueError, match="Unexpected order of git dependency components"):
        parse_dependency_string("git+https://github.com/efriis/myrepo.git#subdirectory=src@branch", None, None, None)

    with pytest.raises(ValueError, match="Unexpected format of git dependency component"):
        parse_dependency_string("git+https://github.com/efriis/myrepo.git@subdirectory=src", None, None, None)


def test_dependency_string_edge_case() -> None:
    test_cases = [
        (
            "git+ssh://a@b",
            {"git": "ssh://a@b", "subdirectory": None, "ref": None},
        ),
        (
            "git+ssh://a@b.com/path",
            {"git": "ssh://a@b.com/path", "subdirectory": None, "ref": None},
        ),
    ]

    for input_string, expected_dep in test_cases:
        parsed_dep = parse_dependency_string(input_string, None, None, None)
        _assert_dependency_equals(parsed_dep, expected_dep)

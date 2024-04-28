SHELL := /bin/bash

PYTEST := poetry run pytest
RUFF := poetry run ruff
MYPY := poetry run mypy
CODESPELL := poetry run codespell --toml pyproject.toml
CHECK_IMPORTS := poetry run python ./scripts/check_imports.py

.PHONY: all format lint test tests integration_tests docker_tests help help_extended pre-commit pre-push check_imports format_diff format_diff_staged lint_diff lint_diff_staged lint_package lint_package_staged lint_tests lint_tests_staged mypy_check mypy_check_staged spell_check format_check lint_check

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/


test tests:
	$(PYTEST) --disable-socket --allow-unit-socket $(TEST_FILE)

integration_test integration_tests:
	$(PYTEST) $(TEST_FILE)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/__package_name_short__ --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=__module_name__
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	$(RUFF) .
	$(RUFF) format $(PYTHON_FILES)
	$(RUFF) --select I $(PYTHON_FILES)
	mkdir -p $(MYPY_CACHE); $(MYPY) $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	$(RUFF) format $(PYTHON_FILES)

spell_check:
	$(CODESPELL)

spell_fix:
	$(CODESPELL) -w

check_imports: $(shell find __module_name__ -name '*.py')
	$(CHECK_IMPORTS) $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'integration_tests            - run integration tests'
	@echo 'docker_tests                 - run tests in a Docker container'
	@echo 'extended_tests               - run extended tests'
	@echo 'pre-commit                    - run pre-commit hooks'
	@echo 'pre-push                     - run pre-commit hooks and tests before pushing changes'
	@echo 'format_diff_staged           - format staged files'
	@echo 'lint_diff_staged             - lint staged files'
	@echo 'lint_package_staged          - lint staged package files'
	@echo 'lint_tests_staged            - lint staged test files'
	@echo 'check_imports_staged         - check staged imports'
	@echo 'format_check                 - check if the code is properly formatted'
	@echo 'lint_check                   - check if the code passes linting'
	@echo 'mypy_check                   - check if the code passes static type checking'
	@echo 'spell_check                  - check if the code is spelled correctly'
	@echo 'help_extended                - display an extended help message'

help_extended:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo '	$(RUFF) format $(PYTHON_FILES)'
	@echo 'lint                         - run linters'
	@echo '	$(RUFF) .'
	@echo '	$(RUFF) format $(PYTHON_FILES)'
	@echo '	$(RUFF) --select I $(PYTHON_FILES)'
	@echo 'test                         - run unit tests'
	@echo '	$(PYTEST) --disable-socket --allow-unit-socket $(TEST_FILE)'
	@echo 'tests                        - run unit tests'
	@echo '	$(PYTEST) $(TEST_FILE)'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'integration_tests            - run integration tests'
	@echo '	$(PYTEST) $(integration_test)'
	@echo 'docker_tests                 - run tests in a Docker container'
	@echo '	$(PYTEST) --docker'
	@echo 'extended_tests               - run extended tests'
	@echo '	$(PYTEST) --extended'
	@echo 'pre-commit                    - run pre-commit hooks'
	@echo '	make pre-commit'
	@echo 'pre-push                     - run pre-commit hooks and tests before pushing changes'
	@echo '	make pre-push'
	@echo 'format_diff_staged           - format staged files'
	@echo '	git diff --relative=libs/partners/__package_name_short__ --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' | xargs $(RUFF) format'
	@echo 'lint_diff_staged             - lint staged files'
	@echo '	git diff --relative=libs/partners/__package_name_short__ --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' | xargs $(RUFF) --select I'
	@echo 'lint_package_staged          - lint staged package files'
	@echo '	$(RUFF) --select I $(shell find __module_name__ -name \*.py)'
	@echo 'lint_tests_staged            - lint staged test files'
	@echo '	$(RUFF) --select I $(shell find tests -name \*.py)'
	@echo 'check_imports_staged         - check staged imports'
	@echo '	git diff --relative=libs/partners/__package_name_short__ --name-only --diff-filter=d master | grep -E '\.py$$' | xargs $(CHECK_IMPORTS)'
	@echo 'format_check                 - check if the code is properly formatted'
	@echo '	$(RUFF) --diff $(PYTHON_FILES)'
	@echo 'lint_check                   - check if the code passes linting'
	@echo '	$(RUFF) --select I $(PYTHON_FILES)'
	@echo 'mypy_check                   - check if the code passes static type checking'
	@echo '	$(MYPY) $(PYTHON_FILES)'
	@echo 'spell_check                  - check if the code is spelled correctly'
	@echo '	$(CODESPELL)'

######################
# GIT HOOKS
######################

.gitignore:
	echo "mypy_cache" >> .gitignore

.pre-commit-config.yaml:
	poetry run pre-commit install

pre-commit:
	poetry run pre-commit run --all-files

pre-push:
	poetry run pre-commit run --all-files
	$(PYTEST)

######################
# TESTS
######################

test_integration:
	$(PYTEST) $(integration_test)

test_extended:
	$(PYTEST) --extended

test_docker:
	$(PYTEST) --docker

######################
# CHECKS
######################

format_check:
	$(RUFF) --diff $(PYTHON_FILES)

lint_check:
	$(RUFF) --select I $(PYTHON_FILES)

mypy_check:
	$(MYPY) $(PYTHON_FILES)

spell_check:
	$(CODESPELL)

######################
# STAGED FILES
######################

format_diff_staged:
	git diff --relative=libs/partners/__package_name_short__ --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' | xargs $(RUFF) format

lint_diff_staged:
	git diff --relative=libs/partners/__package_name_short__ --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$' | xargs $(RUFF) --select I


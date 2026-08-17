# Contributing to the NeatLogs Python SDK

This guide describes the local checks and CI maintenance rules for the
`neatlogs` Python SDK.

## CI overview

The SDK CI workflow is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
It runs on every branch push and every pull request, so a pull request may show
two equivalent CI runs for the same commit.

CI has two responsibilities:

1. A single Python 3.13 lint job checks Black formatting and isort import order.
2. Unit-test jobs run on every supported Python version: 3.10, 3.11, 3.12,
   and 3.13.

The lint job is deliberately separate from the Python matrix. A formatting
failure should be reported once; it should not prevent four otherwise
independent test jobs from reporting their results.

## Local development setup

Install the project and development dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The supported Python range is declared in `pyproject.toml`. Use a Python
version inside that range for local development.

### Extras

`uv sync --extra <name>` installs one framework extra. Note that `--all-extras`
will **not** work: `crewai` and `hermes` cannot coexist in one environment, and
`pyproject.toml` declares them as conflicting under `[tool.uv]`.

```bash
uv sync --extra crewai   # or --extra hermes, never both
```

`hermes-agent` publishes no distribution for Python 3.10, so the `hermes` extra
carries a `python_version >= '3.11'` marker. On 3.10 the extra installs cleanly
and simply contributes nothing.

## Required checks before pushing

Run these commands from the repository root:

```bash
uv run black neatlogs tests
uv run isort neatlogs tests
uv run pytest -q tests/unit
git diff --check
```

Run Black before isort. The repository's Black and isort settings are defined
in `pyproject.toml`, and both tools operate on the complete `neatlogs/` and
`tests/` trees.

Do not format only the files changed in the current branch. CI checks both
complete directories, so an unformatted file elsewhere in the tree will still
fail the pull request.

To check formatting without modifying files:

```bash
uv run black --check neatlogs tests
uv run isort --check-only neatlogs tests
```

## Formatting and logical changes

Black changes layout, quoting, wrapping, and other syntax presentation without
intending to change Python behavior. isort can also reorder or regroup import
statements.

When a formatter upgrade produces a large repository-wide diff:

1. Put the formatter baseline in a dedicated commit.
2. Keep functional changes in separate commits or a separate pull request.
3. Run the complete unit-test suite after formatting.
4. Review import-order changes, especially imports inside functions and modules
   that perform registration or instrumentation during import.
5. Avoid mixing manual refactors into the formatter-baseline commit.

A formatter-only commit should be reproducible by checking out its parent and
running the pinned formatter versions. If it cannot be reproduced that way,
review the additional changes as normal code changes.

## Keeping CI dependencies deterministic

CI uses two small, pinned requirements files:

- [`.github/requirements-lint.txt`](.github/requirements-lint.txt) contains
  Black and isort.
- [`.github/requirements-test.txt`](.github/requirements-test.txt) contains
  direct dependencies needed by the unit-test suite.

These files prevent direct formatter and test dependency upgrades from
changing CI results without a repository change. They reduce dependency drift,
but they are not a complete lock of every transitive dependency installed by
pip.

`uv.lock` remains the source of truth for the resolved versions. Whenever
the lock file changes, check whether any package listed in either CI
requirements file also changed. If it did, update the matching pin in the same
pull request.

After installing dependencies, the resolved version of a package can be
checked with:

```bash
uv pip show <package-name>
```

### Updating Black or isort

When intentionally upgrading a formatter:

1. Update its constraint in `pyproject.toml` if required.
2. Regenerate `uv.lock` with `uv lock`.
3. Copy the resolved version into `.github/requirements-lint.txt`.
4. Run Black and isort over both `neatlogs/` and `tests/`.
5. Commit the generated formatter baseline separately from CI configuration
   and functional changes.
6. Run the required local checks in this guide.

Do not change the formatter pin in CI without applying and reviewing the
corresponding formatter output.

### Adding or updating unit-test dependencies

When a unit test directly needs a new package:

1. Add it to the development dependency group in `pyproject.toml`.
2. Regenerate `uv.lock` with `uv lock`.
3. Add the exact resolved version to `.github/requirements-test.txt`.
4. Verify installation and run `pytest -q tests/unit` in a clean environment.

Only direct unit-test requirements belong in
`.github/requirements-test.txt`. Transitive dependencies continue to be
resolved by pip from the pinned direct dependency set and the installed SDK.
If a transitive dependency causes a reproducible CI regression, constrain it
through the appropriate direct dependency or introduce an explicit pin with a
comment explaining why it is required.

## Clean-environment verification

When changing CI dependencies or diagnosing an installation failure, verify
the same installation sequence used by GitHub Actions in a new virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -e .
python -m pip install -r .github/requirements-test.txt
python -m pip install -r .github/requirements-lint.txt

black --check neatlogs tests
isort --check-only neatlogs tests
pytest -q tests/unit
```

Remove or recreate the virtual environment before repeating this test after a
dependency change. Do not rely on packages left over from unrelated local
development.

## Changing supported Python versions

Python support is defined in two places:

- The Python constraint in `pyproject.toml`.
- The test matrix in `.github/workflows/ci.yml`.

When adding or removing a supported Python version:

1. Update both locations in the same pull request.
2. Confirm the project and CI-only dependencies install on every matrix
   version.
3. Run the unit tests on the new or affected version.
4. Check optional instrumentation dependencies for incompatible Python
   constraints.
5. Keep the lint job on one supported Python version; do not duplicate linting
   across the entire matrix.

## Tests beyond the CI gate

The required CI gate is:

```bash
uv run pytest -q tests/unit
```

Integration tests may require optional framework packages, service-specific
fixtures, credentials, or network access. Run the relevant integration tests
when changing an instrumentation adapter, but do not treat a missing optional
dependency as a regression without reproducing the same test against the base
branch.

Tests must not make real provider API calls unless they are explicitly marked
and documented as live tests.

## Diagnosing CI failures

Use the failing step to decide where to investigate:

- **Black failure:** run `black neatlogs tests`, review the generated diff, and
  rerun Black in check mode.
- **isort failure:** run `isort neatlogs tests`, inspect import-order changes,
  and rerun isort in check mode.
- **One Python version fails:** compare dependency installation and traceback
  output for that matrix version.
- **Every Python version fails identically:** investigate shared source,
  dependency, or test-fixture issues rather than Python-version compatibility.
- **CI began failing without a repository change:** check for an unpinned
  dependency or action. Do not fix it by repeatedly rerunning the workflow.

When investigating an unexpected failure, capture the exact command,
dependency versions, Python version, and first meaningful traceback. A green
formatter check does not prove tests pass, and a green unit suite does not
replace reviewing a large generated diff.

## Pull-request checklist

Before requesting review:

- [ ] Black passes for `neatlogs/` and `tests/`.
- [ ] isort passes for `neatlogs/` and `tests/`.
- [ ] All unit tests pass locally.
- [ ] `git diff --check` passes.
- [ ] CI dependency pins match `uv.lock`.
- [ ] Formatter-only and logical changes are clearly separated.
- [ ] Relevant integration tests were run for instrumentation changes.
- [ ] The pull-request description lists the exact commands and results.
- [ ] Migration, data, security, compatibility, and release impact are stated
      when relevant.

CI-only and formatter-only maintenance does not require an SDK version bump or
release unless packaged runtime behavior or published package metadata also
changes.

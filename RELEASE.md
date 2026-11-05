# Releasing ragcompliance

Nothing in this file has been run against PyPI. The commands build and
check the artefacts locally and then publish them with a token you supply.

## 0. Prerequisites

* A PyPI API token with upload rights for `ragcompliance` (and one for
  TestPyPI if you want a dry run), stored in your shell as
  `PYPI_TOKEN` and `TEST_PYPI_TOKEN`. Never commit them.
* The `main` branch at the commit you are releasing, CI green.
* `pip install build twine` in a clean virtualenv.

## 1. Check the version

The version lives only in `ragcompliance/_version.py`. For this release
candidate it is `1.0.0rc1`. PyPI treats `rc` versions as pre releases, so
`pip install ragcompliance` keeps resolving to `0.1.8` until the final
`1.0.0` is published; users opt in with `pip install --pre ragcompliance`
or `pip install ragcompliance==1.0.0rc1`.

```bash
python -c "import ragcompliance; print(ragcompliance.__version__)"
grep -n "## \[1.0.0rc1\]" CHANGELOG.md
```

## 2. Run the checks

```bash
ruff check ragcompliance tests
```

```bash
mypy
```

```bash
pytest --cov=ragcompliance --cov-fail-under=90
```

```bash
mkdocs build --strict
```

## 3. Build the artefacts

```bash
rm -rf dist build
```

```bash
python -m build
```

```bash
twine check dist/*
```

Expected: `dist/ragcompliance-1.0.0rc1-py3-none-any.whl` and
`dist/ragcompliance-1.0.0rc1.tar.gz`, both `PASSED`. Have a look inside
the sdist to confirm it carries the package, tests, examples, SQL files
and Markdown docs, and no HTML site assets:

```bash
tar tzf dist/ragcompliance-1.0.0rc1.tar.gz | head -50
```

Smoke test the wheel in a fresh environment:

```bash
python -m venv /tmp/rc-smoke && /tmp/rc-smoke/bin/pip install dist/ragcompliance-1.0.0rc1-py3-none-any.whl && /tmp/rc-smoke/bin/ragcompliance selftest --dev-mode
```

## 4. Tag

```bash
git tag -a v1.0.0rc1 -m "ragcompliance 1.0.0rc1"
```

```bash
git push origin v1.0.0rc1
```

## 5. Publish to TestPyPI first (recommended)

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD="$TEST_PYPI_TOKEN" twine upload --repository testpypi dist/*
```

```bash
python -m venv /tmp/rc-test && /tmp/rc-test/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "ragcompliance==1.0.0rc1" && /tmp/rc-test/bin/ragcompliance --version
```

## 6. Publish to PyPI

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_TOKEN" twine upload dist/*
```

Then create the GitHub release from the tag and paste the `1.0.0rc1`
section of `CHANGELOG.md` as the notes. Mark it as a pre release.

## 7. After the release

* Announce the pre release and the migration guide
  (`docs/migration.md`).
* Decide whether to switch GitHub Pages to the mkdocs output. Today
  `docs/` serves the hand written landing site; `mkdocs build` writes the
  reference docs to `site/`. A Pages workflow that runs `mkdocs build`
  and deploys `site/` replaces the landing site, so keep the landing
  content as `docs/index.md` content or a separate page first.
* Open issues for the deferred items in `AUDIT.md` (Sigstore signing,
  asymmetric record signatures).

## Cutting the final 1.0.0

1. Edit `ragcompliance/_version.py` to `1.0.0`.
2. Rename the `[1.0.0rc1]` section in `CHANGELOG.md` to `[1.0.0]` with
   the release date and add anything fixed during the candidate period.
3. Repeat steps 2 to 6 with `v1.0.0`.

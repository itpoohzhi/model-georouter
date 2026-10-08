# Contributing

Solo-maintained project, but tidy contributions are welcome.

## Branches

- `master` — stable, always releasable. Protected: changes only via pull request with green CI.
- `dev` — integration branch. Protected: changes only via pull request with green CI.
- `feature/<short-name>` — work branches, cut from `dev`, merged back via PR.

Flow: `feature/*` → PR → `dev` → PR → `master` (release).

## Checks (must be green before PR)

```bash
python -m pytest
ruff check .
ruff format --check .
```

CI pins `ruff==0.16.10` (see `.github/workflows/ci.yml`). Keep your local ruff
on the same version, otherwise CI and local results may differ:

```bash
pip install "ruff==0.16.10"
```

## Style

- Python 3.11+ standard library only — no runtime dependencies.
- Line length 120 (`pyproject.toml`).
- No secrets, tokens, or local paths in code or tests.

# CLAUDE.md - AI Assistant Guide for Zyra

This document provides essential context for AI assistants (Claude, GitHub Copilot, etc.) working with the Zyra codebase.

## Repository Context: Downstream Mirror

> **Important**: You are likely working in `zyra-project/zyra`, which is a **downstream mirror** of the canonical repository at [`NOAA-GSL/zyra`](https://github.com/NOAA-GSL/zyra).

### How the Mirror Works

```
NOAA-GSL/zyra (upstream)          zyra-project/zyra (downstream)
       │                                    │
       │  ──── sync every 30 min ────►     │
       │        (mirror/main,               │
       │         mirror/staging)            │
       │                                    │
       │                            claude/* branches
       │                            codex/* branches
       │                                    │
       │  ◄──── relay workflow ────        │
       │     (PRs → relay/hh-pr-###)        │
       │                                    │
```

1. **Mirror Sync**: Every 30 minutes, `main` and `staging` from NOAA-GSL/zyra are synced to `mirror/main` and `mirror/staging` branches here. Upstream workflow files are stripped to prevent duplicate CI runs.

2. **AI Agent Branches**: AI assistants (Claude, Codex) create branches with prefixes like `claude/*` or `codex/*` in this downstream repo.

3. **PR Relay**: When a PR targets `mirror/staging` in this repo, an automated workflow:
   - Rebases the changes onto `NOAA-GSL/zyra:staging`
   - Pushes to a `relay/hh-pr-<number>` branch in NOAA-GSL/zyra
   - Creates/updates a corresponding PR in the upstream repo

4. **Conflict Resolution**: The relay auto-resolves conflicts in infrastructure files (Docker, workflows, lock files) and docs by preferring upstream. Other conflicts require manual resolution.

### What This Means for AI Assistants

- **Work here**: Develop on `claude/*` or `codex/*` branches in zyra-project/zyra
- **Code changes → `mirror/staging`**: PRs with code changes should target `mirror/staging` to be relayed upstream
- **Mirror-specific changes → `main`**: PRs for workflows, documentation, or configuration specific to this mirror repo should target `main`
- **Issues/Discussions upstream**: File issues and discussions at [NOAA-GSL/zyra](https://github.com/NOAA-GSL/zyra/issues)
- **Code is mirrored**: The source code in `mirror/*` branches matches upstream

### Branch Reference

| Branch | Purpose | PR Target For |
|--------|---------|---------------|
| `mirror/main` | Read-only mirror of NOAA-GSL/zyra:main | — |
| `mirror/staging` | Read-only mirror of NOAA-GSL/zyra:staging | Code changes (relayed upstream) |
| `main` | Local workflows and docs for this mirror repo | Mirror-specific changes (stays here) |
| `claude/*`, `codex/*` | AI agent working branches | — |

---

## Project Overview

**Zyra** is an open-source Python framework for modular, reproducible data workflows, maintained by NOAA Global Systems Laboratory. It helps import, process, visualize, and export scientific and environmental data.

- **Version**: 0.1.43
- **License**: Apache-2.0
- **Python**: 3.10+
- **Build System**: Poetry
- **Upstream Repository**: https://github.com/NOAA-GSL/zyra
- **Downstream Mirror**: https://github.com/zyra-project/zyra

## Quick Reference Commands

```bash
# Install with dev dependencies AND all extras (required for pre-commit hooks)
poetry install --with dev --extras "all"

# Format and lint (REQUIRED before commits)
poetry run ruff format . && poetry run ruff check .

# Run tests (focused)
poetry run pytest -q -k <pattern>

# Run full test suite
poetry run pytest

# CLI help
poetry run zyra --help

# Pre-commit hooks
poetry run pre-commit install
poetry run pre-commit run -a

# Update OpenAPI snapshot (REQUIRED when CLI/API changes affect the manifest)
./scripts/update_openapi_snapshot.sh
```

## Codebase Architecture

### Directory Structure

```
zyra/
├── src/zyra/              # Main package source
│   ├── cli.py             # Root CLI entry point
│   ├── api_cli.py         # API service CLI
│   ├── pipeline_runner.py # Config-driven pipeline executor
│   ├── manifest.py        # Programmatic manifest API
│   ├── plugins.py         # Plugin registration system
│   ├── connectors/        # Data ingress/egress
│   │   ├── ingest/        # HTTP, S3, FTP, Vimeo downloaders
│   │   ├── egress/        # Export to local/S3/FTP/HTTP/Vimeo
│   │   ├── backends/      # Shared AWS, FTP, HTTP clients
│   │   └── discovery/     # Semantic search & discovery
│   ├── processing/        # GRIB2, NetCDF, GeoTIFF, audio/video
│   ├── visualization/     # Heatmap, contour, animate, interactive
│   ├── transform/         # Metadata ops, enrich, dataset JSON
│   ├── api/               # FastAPI REST service
│   │   ├── routers/       # Domain routers (acquire, process, etc.)
│   │   ├── models/        # Request/response schemas
│   │   └── mcp_tools/     # Model Context Protocol tools
│   ├── swarm/             # Multi-agent orchestration
│   ├── narrate/           # LLM-driven narrative generation
│   ├── wizard/            # Interactive CLI assistant
│   ├── workflow/          # Programmatic workflow execution
│   ├── notebook/          # Jupyter integration
│   ├── utils/             # Shared utilities
│   └── assets/            # Packaged resources (prompts, images)
├── tests/                 # Test suite (20+ categories)
├── samples/               # Sample configs and workflows
│   ├── pipelines/         # YAML/JSON pipeline examples
│   ├── workflows/         # Workflow definition examples
│   └── swarm/             # Multi-agent orchestration examples
├── docs/                  # Documentation
│   └── source/wiki/       # Synced mirror of GitHub Wiki
├── docker/                # Docker configurations
├── pyproject.toml         # Poetry configuration
└── ruff.toml              # Linting/formatting rules
```

### The 8-Stage Pipeline Model

Zyra organizes commands into 8 workflow stages:

```
ACQUIRE → PROCESS → SIMULATE → DECIDE → VISUALIZE → NARRATE → VERIFY → EXPORT
```

| Stage | Aliases | Commands |
|-------|---------|----------|
| acquire | ingest | `http`, `s3`, `ftp`, `vimeo` |
| process | transform | `decode-grib2`, `extract-variable`, `convert-format`, `pad-missing` |
| simulate | - | Simulation under uncertainty (skeleton) |
| decide | optimize | Decision/optimization (skeleton) |
| visualize | render | `heatmap`, `contour`, `animate`, `compose-video`, `interactive` |
| narrate | - | AI-driven storytelling/reporting |
| verify | - | Evaluation and metrics (skeleton) |
| export | disseminate | `local`, `s3`, `ftp`, `post`, `vimeo` |

### Entry Points

```toml
# From pyproject.toml
zyra = "zyra.cli:main"        # Main CLI
zyra-cli = "zyra.api_cli:main" # API CLI wrapper
```

## Development Workflow

### Before Making Changes

1. **Read first**: Always read files before modifying them
2. **Lint first**: Run `poetry run ruff format . && poetry run ruff check .`
3. **Understand context**: Check related tests and documentation

### Making Changes

1. **Small, surgical edits**: Change only what's needed
2. **Keep names stable**: Don't rename identifiers unless required
3. **No secrets**: Never commit credentials or tokens
4. **Avoid absolute paths**: Use `DATA_DIR` env var or `importlib.resources`

### After Changes

```bash
# Format and lint (required)
poetry run ruff format . && poetry run ruff check .

# Run focused tests
poetry run pytest -q -k <pattern>

# Verify CLI still works
poetry run zyra --help
```

### Commit Requirements

- **DCO Sign-off**: All commits must include `Signed-off-by` line
  ```bash
  git commit -s -m "Add feature X"
  ```
- **Imperative mood**: "Add GRIB parser" not "Added GRIB parser"
- **Link issues**: Reference related issues in PR descriptions

## Coding Conventions

### Style Guide

- **Python**: 3.10+, UTF-8, Unix newlines, 4-space indent
- **Names**:
  - Files/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
- **Imports**: Grouped as stdlib → third-party → local
- **Formatting**: Ruff handles everything (replaces Black/Isort/Flake8)

### Ruff Configuration

From `ruff.toml`:
```toml
[lint]
select = ["E", "W", "F", "UP", "I", "PTH", "PT", "B", "SIM"]
ignore = ["E501", "W505"]  # Line length handled by formatter
```

Per-file exceptions exist for tests and visualization modules.

### Testing Conventions

- Tests live in `tests/` with `test_*.py` naming
- Use pytest markers: `cli`, `pipeline`, `redis`, `mcp_ws`, `guardrails`
- Prefer small fixtures and sample assets for data-heavy tests
- Use Hypothesis for property-based testing where appropriate

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/zyra/cli.py` | Main CLI implementation (~1168 lines) |
| `src/zyra/pipeline_runner.py` | YAML/JSON pipeline executor (~866 lines) |
| `src/zyra/manifest.py` | CLI capability discovery |
| `src/zyra/plugins.py` | Plugin registration system |
| `src/zyra/stage_utils.py` | Stage name normalization |
| `pyproject.toml` | Dependencies and build config |
| `ruff.toml` | Linting rules |

## Optional Dependencies (Extras)

Install specific features as needed:

```bash
pip install "zyra[connectors]"     # S3, HTTP, Vimeo transfers
pip install "zyra[processing]"     # GRIB2, NetCDF, GeoTIFF
pip install "zyra[visualization]"  # Cartopy/Matplotlib plotting
pip install "zyra[api]"            # FastAPI service layer
pip install "zyra[llm]"            # LLM providers (OpenAI, Gemini, etc.)
pip install "zyra[wizard]"         # Interactive CLI assistant
pip install "zyra[guardrails]"     # Output validation
pip install "zyra[all]"            # Everything
```

## Plugin System

Register custom commands programmatically:

```python
from zyra import plugins

# Direct registration
plugins.register_command("process", "my_plugin", handler=my_func, description="My plugin")

# Decorator style
@plugins.register_command("visualize", "custom_plot")
def custom_plot(data, output):
    ...
```

Local plugins can be placed in `.zyra/extensions/plugins.py` and are auto-loaded (disable with `ZYRA_DISABLE_LOCAL_PLUGINS=1`).

## Programmatic API

### Workflow Execution

```python
from zyra.workflow.api import Workflow

wf = Workflow.load("samples/workflows/minimal.yml")
result = wf.run(capture=True, stream=False)
assert result.succeeded
```

### Manifest Discovery

```python
from zyra.manifest import Manifest

manifest = Manifest.load(include_plugins=True)
commands = manifest.list_commands(stage="process")
```

### Notebook Integration

```python
from zyra.notebook import create_session

sess = create_session()
sess.process.convert_format(file_or_url="/tmp/in.grib2", output="/tmp/out.tif")
```

## Configuration Files

### Pipeline YAML Format

```yaml
# samples/workflows/minimal.yml
stages:
  - stage: narrate
    command: describe
    args:
      topic: "workflow smoke test"
```

### Swarm Agent YAML Format

```yaml
# samples/swarm/mock_basic.yaml
metadata:
  description: Sample manifest for mock simulate → narrate swarm
agents:
  - id: simulate
    stage: simulate
    outputs:
      - simulated_output
  - id: narrate
    stage: narrate
    depends_on:
      - simulate
    stdin_from: simulated_output
```

## Documentation Resources

| Resource | Location |
|----------|----------|
| Wiki (synced mirror) | `/docs/source/wiki/` |
| Wiki (live) | https://github.com/NOAA-GSL/zyra/wiki |
| API Reference | https://noaa-gsl.github.io/zyra/ |
| Module READMEs | `src/zyra/*/README.md` |
| Sample pipelines | `samples/pipelines/*.yaml` |

### Key Wiki Pages

- [Workflow Stages](https://github.com/NOAA-GSL/zyra/wiki/Workflow-Stages)
- [Stage Examples](https://github.com/NOAA-GSL/zyra/wiki/Stage-Examples)
- [Install & Extras](https://github.com/NOAA-GSL/zyra/wiki/Install-Extras)
- [Pipeline Patterns](https://github.com/NOAA-GSL/zyra/wiki/Pipeline-Patterns)

## Common Tasks

### Adding a New CLI Command

1. Create function in appropriate module (e.g., `processing/`, `visualization/`)
2. Register with Click decorator in CLI
3. Add tests in corresponding `tests/` directory
4. Update module README with usage examples
5. Keep docstrings current for autodoc

### Adding a New Connector

1. Add implementation in `connectors/ingest/` or `connectors/egress/`
2. Use shared backends from `connectors/backends/` where applicable
3. Add integration tests
4. Update `connectors/*/README.md`

### Running the API Server

```bash
# Development server
poetry run uvicorn zyra.api:app --reload

# Or via CLI
poetry run zyra-cli serve
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DATA_DIR` | Base directory for data files |
| `ZYRA_DISABLE_LOCAL_PLUGINS` | Disable auto-loading of local plugins |
| `ZYRA_NOTEBOOK_OVERLAY` | Custom manifest overlay for notebooks |
| `OPENAI_API_KEY` | OpenAI API key for LLM features |
| `GOOGLE_API_KEY` | Google API key for Gemini |
| `VERTEX_PROJECT` | Google Cloud project for Vertex AI |

## Troubleshooting

### Import Errors

Many features require optional dependencies. Check the extras section and install needed packages:
```bash
pip install "zyra[processing]"  # For GRIB/NetCDF
pip install "zyra[visualization]"  # For plotting
```

### FFmpeg Required

Audio/video processing requires FFmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Pre-commit Hook Failures

```bash
# Fix formatting issues
poetry run ruff format .

# Fix linting issues (some auto-fixable)
poetry run ruff check --fix .
```

### Manifest Generation Failures

The `zyra-generate-manifest` pre-commit hook requires all optional dependencies to be installed. If the manifest is missing commands (e.g., `narrate.json` deleted), install all extras:

```bash
poetry install --extras "all"
poetry run zyra generate-manifest
```

Without all extras, the generator cannot introspect commands that depend on optional packages (like `pydantic` for narrate schemas).

## Agent Checklist

Before returning code changes, verify:

- [ ] Ran `poetry run ruff format . && poetry run ruff check .` - clean output
- [ ] Ran targeted tests (`poetry run pytest -q -k <pattern>`) - passing
- [ ] If CLI/API commands changed, ran `./scripts/update_openapi_snapshot.sh`
- [ ] Limited scope to requested changes only
- [ ] Updated documentation if needed
- [ ] Called out assumptions and external requirements (e.g., FFmpeg)
- [ ] No secrets or credentials committed

### For PRs in the Downstream Mirror

- [ ] Working on a `claude/*` or `codex/*` branch
- [ ] PR targets the correct branch:
  - `mirror/staging` for code changes (will be relayed upstream to NOAA-GSL/zyra)
  - `main` for mirror-specific docs/workflows (stays in zyra-project/zyra)
- [ ] Any issues or feature requests filed at [NOAA-GSL/zyra](https://github.com/NOAA-GSL/zyra/issues)

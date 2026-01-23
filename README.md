# Zyra Mirror (Downstream)

This repository contains workflows that keep a copy of the upstream
[`NOAA-GSL/zyra`](https://github.com/NOAA-GSL/zyra) repository in this repo,
while also supporting **local branches** and **relay of PRs** back into the
canonical org repo.

⚠️ **Note:** This is *not* the canonical repository. Please go to the
[NOAA-GSL organization repo](https://github.com/NOAA-GSL/zyra) for issues,
pull requests, releases, and active development.

---

## How it works

### 1. Mirror Sync
- Runs every 30 minutes (`cron`) or can be triggered manually via the Actions tab.
- Clones the upstream repo in `--mirror` mode.
- **Strips all upstream workflow files** (`.github/workflows/**`) so NOAA-GSL’s Actions
  (like `sync-to-wiki` or release publishers) do **not** run here.
- Re-adds remotes and force-pushes only the requested branches (`main`, `staging` by default)
  into `mirror/*` branches.
- By default, **does not push tags**, since tags often trigger release workflows.
  You can override this manually.
- Skips pushes if the branch SHA hasn’t changed (no-op runs).

### 2. Local Branches
- Codex (or developers using the provided helper) create branches in this repo under the prefix `codex/*`.
- PRs are opened here with `base = mirror/staging`.

### 3. Relay Workflow
- Listens to PRs in this repo targeting `mirror/staging`.
- Rebases the PR head onto `NOAA-GSL/zyra:staging`.
- Pushes to a deterministic branch in the org repo:
  `relay/hh-pr-<number>`.
- Creates or updates a PR in the org repo with base = `staging`.
- Closes the org PR automatically if the HH PR is closed.
- Requires the `SYNC_PAT_ORG` secret with **push + PR rights** in the upstream org.

---

## Example Flow

1. Mirror sync updates `mirror/staging` from upstream.
2. Developer runs `new-branch.sh codex/my-feature` to create a branch.
3. PR is opened in **zyra-project/zyra** with base = `mirror/staging`.
4. Relay bot mirrors the PR to **NOAA-GSL/zyra:staging** as `relay/hh-pr-###`.

---

## Why mirror branches?

- Keeps upstream code visible for development, comparison, or testing.
- Protects this repo’s `main` branch, workflows, and custom content from being overwritten.
- Provides a clean staging area (`mirror/staging`) for Codex contributions.
- Ensures all merges ultimately land in **NOAA-GSL/zyra**.

---

## Important details

- **Downstream safety**:  
  Only `mirror/*` branches are overwritten. Your local `main` and any `codex/*`
  branches remain untouched.
- **Upstream workflows removed**:  
  Ensures redundant or unsafe workflows (wiki sync, releases) never execute here.
- **Tags default OFF**:  
  Tags are skipped by default to avoid publishing releases.
- **Other workflows in this repo**:  
  If you have additional workflows in this repo, and you don’t want them to fire
  on mirror branch updates, add:
  ```yaml
  on:
    push:
      branches-ignore:
        - 'mirror/**'
  ```
---

## Inputs

When running manually (`workflow_dispatch`), you can customize:
- `branches`: space-separated list of branches to mirror (`main staging` by default).
- `push_tags`: `true`/`false` (default `false`).
- `force`: `true`/`false` (default `true`).

### Example

Mirror `main` only, with tags, and allow force push:
```
branches: "main"
push_tags: "true"
force: "true"
```


This will create/update `mirror/main` and mirror all upstream tags.

---

## Branch Protections

- **NOAA-GSL/zyra**  
  - Protect `main`, `staging`  
  - Require PRs, checks, CODEOWNERS

- **zyra-project/zyra**  
  - Protect `mirror/main`, `mirror/staging` (allow force push only for bot)  
  - Leave `codex/*` open for Codex automation  
  - Protect `main` for local workflows and docs

---

### TL;DR
- ✅ Keeps NOAA-GSL code mirrored here under `mirror/*`.  
- ✅ Provides a safe place for Codex (and helpers) to open PRs.  
- ✅ Relays Codex PRs upstream into NOAA-GSL/zyra:staging.  
- ✅ Prevents running unwanted upstream Actions.  
- ✅ Protects your own workflows and content.  
- ✅ Gives you full control over what branches/tags are synced.

---

## Where to contribute

If you want to contribute code, file issues, or discuss features, use the **upstream repo**:  
👉 [NOAA-GSL/zyra](https://github.com/NOAA-GSL/zyra)

This downstream is read-only, maintained by automation.

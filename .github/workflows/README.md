# Sync from NOAA-GSL → mirror/*

This repository contains a workflow that keeps a copy of the upstream  
[`NOAA-GSL/zyra`](https://github.com/NOAA-GSL/zyra) repository in this repo, but **safely isolated under `mirror/*` branches**.

⚠️ For contribution rules (branching, PRs, relay), see [AGENTS.md](../../AGENTS.md).  
⚠️ For overall project context, see the main [README.md](../../README.md).

---

## How it works

- Runs every 30 minutes (`cron`) or can be triggered manually via the Actions tab.
- Clones the upstream repo in `--mirror` mode.
- **Strips all upstream workflow files** (`.github/workflows/**`).  
  This prevents accidental triggering of upstream jobs like `sync-to-wiki` or release publishing.
- Re-adds remotes and force-pushes only the requested branches (`main`, `staging` by default) into `mirror/*` branches.
- By default, **does not push tags**, since tags often trigger release workflows. You can override this manually.
- Skips pushes if the branch SHA hasn’t changed (no-op runs).
- Uses a concurrency group so that only one sync job runs at a time.

---

## Why mirror branches?

- Keeps upstream code visible for development, comparison, or testing.
- Protects this repo’s `main` branch, workflows, and custom content from being overwritten.
- Allows controlled selective sync (e.g. only certain branches, tags optional).

---

## Important details

- **Downstream safety**:  
  Only `mirror/*` branches are overwritten. Your local `main` (with the `sync-upstream` workflow itself) and other branches remain untouched.
- **Upstream workflows removed**:  
  Ensures redundant or unsafe workflows (wiki sync, releases) never execute here.
- **Tags default OFF**:  
  Tags are skipped by default to avoid publishing releases. You can enable by setting `push_tags: true` when triggering manually.
- **Other workflows in this repo**:  
  If you have additional workflows in this repo, and you don’t want them to fire on mirror branch updates, add the following to their triggers:
  ```yaml
  on:
    push:
      branches-ignore:
        - 'mirror/**'
  ```

## Inputs

When running manually (`workflow_dispatch`), you can customize:

- `branches` (default: `"main staging"`)  
  Space-separated list of branches to mirror from upstream.  
- `push_tags` (default: `"false"`)  
  Mirror tags too? Set to `true` to include them.  
- `force` (default: `"true"`)  
  Force-push? Safe here since only `mirror/*` is affected.  

---

## Example

Mirror `main` only, with tags, and allow force push:

```
branches: "main"
push_tags: "true"
force: "true"
```


This will create/update `mirror/main` and mirror all upstream tags.

---

## TL;DR
- ✅ Keeps NOAA-GSL code mirrored here.  
- ✅ Prevents running their Actions.  
- ✅ Protects your own workflows and content.  
- ✅ Gives you full control over what branches/tags are synced.  


---

## See also
- [README.md](../../README.md) — Overview of downstream repo purpose & relay workflow  
- [AGENTS.md](../../AGENTS.md) — Contribution guide for Codex branches, PR flow, and relay rules 

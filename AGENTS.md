# AGENTS.md — Contribution Guide (Downstream Mirror + Codex + Relay)

> **Who is this for?**  
> Engineers, docs writers, and automation “agents” who work with the Zyra codebase.  
> **Important:** This repo is a **downstream mirror + contribution relay**.  
> The **source of truth is upstream** at [NOAA-GSL/zyra](https://github.com/NOAA-GSL/zyra).  
> All PRs opened here will be relayed into the org repo.

---

## TL;DR

- ✅ **Create new branches here** (`zyra-project/zyra`) under `codex/*` or `feat/*`.
- ✅ Open PRs **against `mirror/staging`** in this repo.
- 🤖 The relay workflow will **rebase and open/update a PR upstream** in `NOAA-GSL/zyra:staging`.
- 🚫 **Never** commit to `mirror/*` — they are overwritten by automation.
- ✅ `main` in this repo is for docs/workflows, not code.
- 🔗 **Upstream issues are synced here** — look for the `upstream-sync` label.  

---

## Repositories & Roles

- **Upstream (canonical):** `NOAA-GSL/zyra`
  - All *final* merges happen here.
  - Releases, tags, and CODEOWNERS reviews live here.
  - Issues are the source of truth but are **synced to downstream**.  

- **Downstream (this repo — HacksHaven/zyra):**  
  - `mirror/main`, `mirror/staging`: read-only, force-pushed from upstream.  
  - `codex/*`: for local AI/automation or contributor branches.  
  - `main`: houses workflows/docs; protected.  
  - Relay workflow ensures PRs here flow back to upstream.

---

## Branch Policy

### Long-lived branches
- **Upstream**
  - `main` → default, production-ready.
  - `staging` → integration/pre-release line.

- **Downstream**
  - `mirror/*` → read-only mirrors of upstream (`main`, `staging`).
  - `main` → local workflows/docs, not mirrored.

### Short-lived branches
- **Create in downstream (`HacksHaven/zyra`)**, not upstream.  
- Prefixes:
```
codex/<slug>
feat/<slug>-<issue#>
fix/<slug>-<issue#>
docs/<slug>
chore/<slug>
```

Examples:
- `codex/add-cli-tests`
- `feat/new-login-flow-742`
- `fix/null-ref-803`

---

## Where to Create New Code

1. **With write access here (Codex or human):**
 - Base your branch from `mirror/staging`.
 - Create `codex/*` (or `feat/*`) branch.
 - Open a PR with `base = mirror/staging`.

2. **Relay workflow does the rest:**
 - Rebases your branch onto upstream `staging`.
 - Pushes branch `relay/hh-pr-<number>` in org repo.
 - Creates or updates a PR upstream.

3. **Reviews and merges happen upstream** (`NOAA-GSL/zyra`).  
 Your downstream PR will stay open but should not be merged manually.

---

## Creating & Maintaining a Branch

```bash
# 1) Clone downstream
git clone https://github.com/HacksHaven/zyra.git
cd zyra

# 2) Base from mirror/staging
git checkout mirror/staging

# 3) Create your feature branch
git checkout -b codex/add-cli-tests

# ... commit code ...

# 4) Keep fresh – rebase regularly
git fetch origin
git rebase origin/mirror/staging

# 5) Push branch
git push -u origin codex/add-cli-tests
```

Follow Conventional Commits:

```
<type>(scope?): short summary

Body explains what & why.
Reference issues: Closes #123
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `perf`, `test`, `build`, `ci`.

---

## Pull Requests (Downstream → Upstream)

- **Open in zyra-project/zyra**, base = `mirror/staging`.
- Relay workflow creates/updates **org PR** automatically.  
- Title/body will reference the original PR for traceability.
- Do **not** merge downstream PRs manually. Close them if not needed.  

**Upstream PR checklist (via relay):**
- ✅ Tests & linting pass  
- ✅ Docs updated if needed  
- ✅ Linked to issues (`Closes #NNN`)  
- ✅ Clear title & context  

---

## CI, Checks, and Environments

- CI runs upstream in `NOAA-GSL/zyra`.  
- Downstream workflows are minimal: mirror, relay, and docs maintenance.  

---

## Releases & Tags

- Releases and tags are **managed upstream only**.  
- Tags are not mirrored here by default.  

---

## Security & Secrets

- Never commit credentials.  
- Use `.env.example` for configuration.  
- If a secret is leaked, rotate it and inform maintainers.  

---

## Interacting with Mirror Branches

- **Never** push to `mirror/*`.
- They are force-updated from upstream.
- To work, branch off `mirror/staging`, never modify it directly.

---

## Working with Synced Issues

Upstream issues from `NOAA-GSL/zyra` are automatically mirrored here.

**How to identify synced issues:**
- They have the `upstream-sync` label.
- The issue body contains a link to the original upstream issue.

**What you can do:**
- **View** upstream issues without leaving this repo.
- **Close/reopen** synced issues — the status change syncs back to upstream.
- **Reference** them in commits and PRs (e.g., `Fixes #123`).

**What you should NOT do:**
- Remove the `upstream-sync` label (breaks tracking).
- Edit the issue body's upstream link marker.

**Creating new issues:**
- New issues should be filed upstream at [NOAA-GSL/zyra](https://github.com/NOAA-GSL/zyra/issues).
- They will be mirrored here on the next sync (every 30 min).

---

## FAQ

**Q: Can I push directly to `mirror/*`?**  
A: No. They are overwritten by automation.  

**Q: Where should I open PRs?**  
A: Here in `zyra-project/zyra`, base = `mirror/staging`. The relay bot will create the org PR.  

**Q: Should I open PRs directly in NOAA-GSL/zyra?**  
A: Only maintainers do that. Normal flow is: downstream PR → relay → upstream PR.  

**Q: What if I need a downstream-only hotfix?**
A: Use a temporary non-mirror branch (`hotfix/<slug>`). Coordinate with maintainers; these should be short-lived.

**Q: Where should I file new issues?**
A: File new issues upstream at [NOAA-GSL/zyra](https://github.com/NOAA-GSL/zyra/issues). They will sync here automatically.

**Q: Can I close an upstream issue from here?**
A: Yes! Close the synced issue here, and the status will sync back to upstream.

---

## Contact & Ownership

- Primary maintenance: Upstream maintainers (`NOAA-GSL/zyra`).  
- Downstream mirror & relay automation: GitHub Actions in this repo.  

# Contributing to Zyra

Thanks for your interest in contributing!
This project thrives on community contributions, and we welcome improvements of all kinds.

> **Note:** This repository (`zyra-project/zyra`) is a **downstream mirror and contribution relay**.
> The **canonical source of truth is [`NOAA-GSL/zyra`](https://github.com/NOAA-GSL/zyra)**.
> Pull requests opened here are automatically relayed upstream. See the workflow below.

---

## License and Contributor Terms

- Zyra is licensed under the Apache License, Version 2.0. See `LICENSE` at the repository root.
- By submitting a pull request, issue suggestion, or any code/documentation/artwork ("Contribution"),
  you agree to license your Contribution under the Apache License, Version 2.0, and you represent that you have the
  right to do so.
- Do not contribute code or assets you don't have rights to. If you include third‑party code or data,
  ensure it is compatible with Apache 2.0 and include proper attribution as required by the original license.
- No CLA is required at this time; contributions are accepted under the project's Apache License terms.
- This project enforces the Developer Certificate of Origin (DCO) via the GitHub DCO app approved by NOAA. All commits must include a Signed-off-by trailer.

If you have questions about licensing or attribution, please open an issue before submitting your PR.

---

## Branching Workflow

This repo uses **read-only mirror branches** synced from upstream, plus short-lived contributor branches that relay into the canonical repo.

### Branch types

- **`mirror/main`** → Read-only mirror of `NOAA-GSL/zyra:main`. Do **not** commit here; it is overwritten by automation.
- **`mirror/staging`** → Read-only mirror of `NOAA-GSL/zyra:staging`. The base for all contributor branches and PRs.
- **`main`** → Local workflows and docs for this relay repo only; not mirrored.

### Rules

1. **Feature Development**
   - Branch off `mirror/staging` (or `mirror/main` for hotfixes):
     ```bash
     git fetch origin
     git checkout -b feat/my-feature-<issue#> mirror/staging
     ```
   - Use one of these prefixes for your branch name:
     | Prefix | Use for |
     |---|---|
     | `feat/<slug>-<issue#>` | New features |
     | `fix/<slug>-<issue#>` | Bug fixes |
     | `docs/<slug>` | Documentation only |
     | `chore/<slug>` | Maintenance, CI, deps |
     | `codex/<slug>` | AI/automation branches |

   - Open a Pull Request targeting **`mirror/staging`** in this repo.

2. **Relay to Upstream**
   - Once your PR is opened here, the relay workflow automatically rebases it onto `NOAA-GSL/zyra:staging` and opens or updates a corresponding PR upstream.
   - The upstream PR will be linked in a comment on your PR here.
   - Review and approval happens on the upstream PR at `NOAA-GSL/zyra`.
   - Closing your PR here will automatically close the relayed upstream PR.

3. **Never commit directly to `mirror/*`**
   - These branches are force-pushed by automation every 30 minutes and any direct commits will be lost.

---

## Issues

Issues are **synced bidirectionally** between `NOAA-GSL/zyra` (upstream) and this repo.

- **Bug reports, feature requests, and workflow gaps** can be filed here or directly on [`NOAA-GSL/zyra`](https://github.com/NOAA-GSL/zyra/issues). Either way they will appear in both places.
- Upstream issues synced here carry the `upstream-sync` label and link back to the original.
- Closing or reopening a synced issue here will update the upstream issue on the next sync.
- **Note:** Only the 100 most recently updated upstream issues are synced per run. Older issues may not appear here; file those directly on the upstream repo.

### Filing Bug Reports
- Use the `🐞 Bug Report` template (`.github/ISSUE_TEMPLATE/bug_report.md`).
- Provide clear steps to reproduce, expected vs. actual behavior, and environment details.

### Filing Feature Requests
- Use the `✨ Feature Request` template (`.github/ISSUE_TEMPLATE/feature_request.md`).
- Describe the feature, motivation, proposed solution, and alternatives.
- Use this template only for enhancements that do **not** map directly to CLI commands.

### Filing Workflow Gap Issues
- Use the `⚡ Workflow Gap / Missing Command` template (`.github/ISSUE_TEMPLATE/workflow_gap.md`).
- Clearly describe:
  - Which CLI commands exist today
  - What is missing
  - Why the feature is needed
- The template will guide you to include an implementation plan and examples.

### Filing Task Issues (Maintenance / Chores)
- Use the `🧹 Task` template (`.github/ISSUE_TEMPLATE/task.md`).
- Use this for non-functional work such as refactors, dependency updates, CI or docs maintenance, code hygiene, and cleanup tasks.
- Do not use for bugs or new features; if the work changes CLI semantics or adds commands, prefer the appropriate Bug/Feature/Workflow Gap template.
- Please include:
  - A concise scope statement (what is and is not in scope)
  - Acceptance criteria (clear, testable completion conditions)
  - Impact/risk notes (blast radius, rollback considerations)
  - Validation steps (how reviewers can verify the task)
  - Links to related issues/PRs

### Submitting PRs for Workflow Gaps
- All PRs that add CLI functionality should link to the related Workflow Gap issue.
- The PR template (`.github/PULL_REQUEST_TEMPLATE.md`) includes a checklist:
  - Add tests
  - Write comprehensive **docstrings** (for auto-generated docs)
  - Include examples in workflows
- Ensure all boxes are checked before requesting review.

By following these templates, contributors help keep Zyra's CLI aligned with real workflows and ensure documentation stays accurate and reproducible.

---

## Code Style

- Python 3.10+ required.
- Follow [PEP8](https://peps.python.org/pep-0008/).
- Run `ruff` and `pytest` locally before opening a PR.

---

## Testing

1. Install dev dependencies:
   ```bash
   poetry install
   ```
2. Run tests:
   ```bash
   pytest
   ```

---

## Pull Requests

- Make sure your branch is up-to-date with `mirror/staging`.
- Include descriptive commit messages with a `Signed-off-by` trailer (see DCO section below).
- Request a review from at least one maintainer.
- Link the related issue (Bug/Feature/Workflow Gap/Task) in the PR description.
- Final review and merge happens on the upstream PR relayed to [`NOAA-GSL/zyra`](https://github.com/NOAA-GSL/zyra).

---

## Releases

Releases are tagged and published from `NOAA-GSL/zyra:main`. This relay repo does not publish releases. To follow release progress, watch the upstream repository.

---

## Developer Certificate of Origin (DCO)

This project uses the DCO to ensure that contributors have the right to submit their work. The full text is included in the `DCO` file at the repository root.

All commits must include a Signed-off-by line matching your Git author information. Use the `-s` flag when committing to add this automatically:

```bash
git commit -s -m "Add feature X"
```

If you forgot to sign off, amend the most recent commit:

```bash
git commit --amend -s --no-edit
```

For multiple commits, you can interactively rebase and sign each commit:

```bash
git rebase -i <base-branch>
# then for each commit: edit -> git commit --amend -s --no-edit -> git rebase --continue
```

Notes
- The Signed-off-by line must include your real name and a reachable email, for example:
  `Signed-off-by: Jane Doe <jane.doe@example.com>`
- Ensure your `git config user.name` and `user.email` are correct.
- Co-authored commits require a Signed-off-by for each author.
- The DCO check will run on pull requests; failures include instructions on how to fix your commits.

Enable global sign-off (recommended)

To automatically include a DCO sign-off on every commit from your machine, enable global sign-off:

```bash
git config --global format.signoff true
```

This works with most Git clients and IDEs (including VS Code) and reduces the chance of missing a sign-off.

---

Thanks again for contributing!

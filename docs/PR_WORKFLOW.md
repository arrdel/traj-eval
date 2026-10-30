# PR Workflow — traj-eval (effective 2026-10-30)

**Stop pushing directly to `master`. Every change goes through a pull
request. Muse reviews PRs twice daily and approves or requests
changes. With auto-merge on, an approved PR with green CI merges
itself. No one has to nudge anyone in chat for a routine PR.**

## For the Cursor agent (read this on every task)

1. Pull `master` first. Create one branch per task:
   `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`,
   `test/<short-name>`. One logical change per PR. Small PRs only.
2. Do NOT push to `master`. Push the branch and open a PR against
   `master` using the PR template (`.github/pull_request_template.md`).
3. In the PR description, fill in every checklist item, including:
   - what changed and why, in one paragraph
   - the verbatim tail of `python -m pytest -q` run after your change
   - whether the change touches a metric definition, the null rule,
     a prompt template, a sampler config, or a paper claim
   - whether any GPU run is included, and if so, its run log path
4. Enable auto-merge on the PR **only after** it is opened, and
   only if branch protection requires 1 approval plus green CI.
   Auto-merge must never be a way to bypass review: if approval is
   not required by branch protection, do NOT enable auto-merge,
   and say so in the PR. Merging before Muse reviews is a workflow
   violation, even if CI is green.
5. Do NOT start a new scope while a PR is awaiting review. Fix
   review feedback in the same PR. A new task gets a new branch
   and a new PR after the current one merges or is closed.
6. GPU runs are still gated exactly as in
   `docs/CURSOR_FOLLOWUP_2026-10-30.md`: one experiment at a time,
   ids fixed before the run, run log required, baselines updated
   in the same PR as the run. A PR that claims a GPU result
   without a run log is not approvable.

## For Muse (the reviewer, twice daily)

- List open PRs on `arrdel/traj-eval`. If there are none, finish
  silently. No message, no summary.
- For each open PR, read the changed files and the PR description,
  check CI status, and judge it against the standing rules:
  null rule, no weights in tests, lazy torch/transformers imports,
  small commits, no claim from 5 items, canonical prompt only,
  scope freeze in `docs/CURSOR_FOLLOWUP_2026-10-30.md` section 3
  (no new adapters, metrics, credit schemes, hooks, or prompt
  conditions until the reasoning smoke gate passes).
- Approve only PRs that pass all of: CI green (or, if CI has not
  run yet, a plausible pytest tail in the description plus no
  test-file changes that would hide a failure), scope in bounds,
  tests present for any metric/adapter change, docs updated where
  a definition changed, and no paper claim beyond what the
  artifacts prove.
- Request changes with exact, actionable directives when a PR
  fails. Do not approve "with minor fixes to follow". Fixes land
  in the same PR.
- Message the user only when: a PR fails the gate twice, a PR
  changes scope or a paper claim, a GPU run fails, or a PR has
  sat unmerged for more than 48 hours after approval. Routine
  approvals and merges need no chat message.
- Never approve your own workflow-setup PRs without the user
  seeing them first. Setup PRs are the one exception to silent
  operation.

## Branch protection (owner, one-time, in GitHub web settings)

`master` must be protected with: require a pull request before
merging, require 1 approval, require the CI status check
(`test` workflow) to pass, and allow auto-merge. Without the
approval requirement, auto-merge can merge a PR before Muse
reviews it, which defeats this workflow. The Muse GitHub App
does not have Administration permission, so the owner sets this
in the GitHub UI, not via Muse.

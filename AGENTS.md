# Agent guidance

Follow [the contribution workflow](docs/how-to-contribute.md): every change starts from
a GitHub issue, uses the issue number as its branch name, includes proportionate tests,
and reaches a shared branch through a pull request.

## Perception encoder experiment line

Perception-encoder research uses a dedicated integration line so exploratory code and
results do not enter `develop` before the preregistered comparisons are complete.

- `exp/perception-encoder` is the long-lived integration branch. It starts from
  `develop` and may periodically merge `develop` forward, but it is never merged
  wholesale back into `develop`.
- Each scoped task has a GitHub issue and branch `exp/<issue-number>`. Create the branch
  from the latest `origin/exp/perception-encoder`, not directly from `develop`.
- Every task issue and frozen run manifest must link the governing specs files at one
  exact `amadou-6e/specs` commit SHA, never at a moving branch. A methodology revision
  requires a recorded deviation plus new preregistration, dataset, and run identities;
  it does not silently alter an active experiment.
- Task pull requests target `exp/perception-encoder`. Use a squash merge so each task
  has one reviewable integration commit. Reference the issue and record its result,
  artifacts, validation, and disposition (`promote`, `retain`, or `reject`). Because
  the base is not the default branch, close the issue explicitly after its integration
  result is accepted; do not rely on a `Closes` keyword.
- Dependent task branches start only after their prerequisites are merged. If a task
  branch is already published, merge the current integration branch into it before
  continuing; do not silently work from a stale experimental base.
- Keep task ownership aligned with
  [the perception-encoder work plan](docs/perception-encoder-experiment-work-plan.md).
  Avoid concurrent edits to another active task's owned modules.
- Commit source, tests, resolved configurations, compact reports, and artifact hashes.
  Do not commit model weights, generated corpora, caches, virtual environments, or raw
  experiment stores.

Promotion is selective:

1. Create a promotion issue and `develop/<promotion-issue>` from current `develop`.
2. Cherry-pick accepted squash commits from `exp/perception-encoder` in dependency
   order. Include required foundation commits; never cherry-pick an integration merge
   commit.
3. Resolve conflicts against current `develop`, run the full affected test suites, and
   open a pull request to `develop` that lists every source experiment issue and SHA.
4. Rejected or inconclusive experiments remain only on the integration line. Results
   may be promoted without promoting a losing implementation when the report is useful.

Specifications under `specs/` are maintained in their own repository. Specification
changes and evidence must be committed and pushed there according to `specs/AGENTS.md`;
do not leave them only in the ignored implementation checkout.

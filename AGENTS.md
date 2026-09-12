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


## Autonomous experiment continuation

When the user authorizes an experiment campaign or says to continue until its
objective is resolved, treat that authorization as covering the remaining
in-scope work, not just the next diagnostic.

- Continue the implementation -> validation -> experiment -> assessment -> next
  evidence-directed action loop within the authorized scope and compute budget.
  Do not ask for another "go ahead" at each routine milestone.
- A completed diagnostic, passing test suite, opened PR, weak score, or
  inconclusive result is not by itself a stopping point. Report it in a progress
  update, identify the next justified action, and execute it when authorized.
  Do not end a turn with only a proposed next step when that step can be done.
- Quality failures direct the next bounded experiment. Validity failures require
  fixing the affected path and excluding invalid evidence before continuing.
  Neither permits lowering acceptance criteria, reusing a contaminated test set,
  changing a frozen run silently, or claiming success from weak results.
- Before each follow-up, check scope, remaining budget, dependencies and protocol.
  Record and freeze required revisions and fresh identities before execution.
  Existing issue/branch/spec/PR requirements still apply. A review dependency may
  be bypassed only by explicit user authorization to stack or execute unmerged
  work; this never authorizes merging or promotion.
- Maintain a durable checkpoint in the tracking issue or campaign progress file:
  objective, governing spec SHA, branch/source SHA, completed runs and artifacts,
  assessment, next concrete action, active process/run IDs, authorized budget
  and recorded usage, and any unresolved approval boundary. Update it at meaningful
  milestones and before a planned handoff. After interruption, inspect this
  checkpoint and existing processes/artifacts before resuming; do not duplicate
  runs or restart the campaign from scratch.
- Pause only when the campaign objective is actually complete, the user pauses
  or redirects it, an explicit approval is needed, the authorized budget is
  exhausted, or a genuine blocker prevents further meaningful in-scope work.
  Do not infer approval of a proposed larger budget from a generic "continue".
  Try safe alternatives and complete unaffected work before declaring a blocker.
- At an unavoidable stop, state what completed, the precise stopping condition,
  the next action, and the minimum decision or external change needed. Do not
  present ordinary poor quality as a permission blocker or invent endless
  follow-ups after the agreed objective is met.

This is an operating rule for the agent, not a background scheduler. Platform
interruptions, tool permissions and session limits still apply; checkpointing
must make the next authorized resume straightforward.

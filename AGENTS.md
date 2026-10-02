# Agent guidance

Follow [the contribution workflow](docs/how-to-contribute.md) for issue-linked
branches, validation, and pull requests.

Specifications are maintained in the separate repository checked out at the
top-level `specs/` folder of the shared local workspace. Do specification work
in that visible checkout, never in an auxiliary specs worktree, so the user can
inspect draft files at their canonical paths. Follow `specs/AGENTS.md` for the
specification repository's own workflow.

Before switching the `specs/` branch, inspect its status and preserve existing
edits. If that checkout cannot be switched safely, ask the user rather than
moving work to a hidden worktree or overwriting their files. Commit and push
completed specification changes in the specs repository; do not leave them only
in this implementation checkout.

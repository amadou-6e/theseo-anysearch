# How to contribute

## Track work in GitHub Issues

Every code change must start with a GitHub issue in this repository. The issue is the canonical place for the problem statement, scope, acceptance criteria, and discussion. Do not add new Markdown tickets under `spec/tickets`.

Use the GitHub issue number in the branch name. For example, work for issue `#42` uses a branch such as `fix/42`.

## Branch naming convention

Use exactly one of these branch prefixes:

| Prefix | Use |
| --- | --- |
| `feat/<issue-number>` | New user-facing behavior or capability |
| `fix/<issue-number>` | Bug fixes and regressions |
| `exp/<issue-number>` | Time-boxed experiments, benchmarks, or research spikes |
| `develop/<issue-number>` | Repository integration or development-line maintenance |
| `master/<issue-number>` | Release-line or production hotfix work |

Examples:

```text
feat/42
fix/57
exp/63
develop/71
master/88
```

Do not use personal or tool-specific prefixes such as `agent/`. Do not replace the issue number with a description. A branch should address one primary issue.

## Contribution workflow

1. Search GitHub Issues to avoid creating a duplicate.
2. Create or select the issue that defines the work.
3. Branch from the appropriate base using `<prefix>/<issue-number>`.
4. Keep the change focused on that issue and add proportionate tests.
5. Commit with a concise description of the completed change.
6. Push the branch and open a pull request.
7. Link the issue in the pull request body with `Closes #<issue-number>` when merging should close it.
8. State what changed, why it changed, and how it was validated.

## Base branches

Normal feature, fix, and experiment work targets `develop`. Use `develop/<issue-number>` or `master/<issue-number>` only when the issue explicitly concerns those integration or release lines. Changes reach `master` through the repository's release process unless an approved hotfix issue says otherwise.

# Curriculum-only task evaluation (#442)

For the next large-world curriculum task, remove the redundant fixed initial-stage
batch by setting `evaluation.enabled: false`. Keep
`evaluation.waypoint_curriculum.enabled: true`, frequency 5, three episodes per
visited stage, and the existing advancement/retention thresholds. The regular
`episodes: 10` setting is then unused: no regular callback, worker pool, or scheduled
batch is created. Curriculum checks still collect their own episodes through the
existing local policy evaluation path.

This is a future methodology deviation governed by the
[pilot spec at a94227bc4ee484287a026f89ec6cd47d5ca16d26](https://github.com/amadou-6e/specs/blob/a94227bc4ee484287a026f89ec6cd47d5ca16d26/projects/theseo-anysearch/python/perception-encoder-pilots.md).
Run `5034f160`, its frozen YAML, dataset, and reports remain unchanged. A future
run needs a new preregistration and run/dataset identities before execution; this
change does not authorize another training run.

Report success per stage rather than treating the old first-stage score as
large-world performance. Evaluation-dependent training early stopping is rejected
when regular evaluation is disabled, rather than silently ceasing to work.
Regular trajectory saving receives no episodes in this mode. Saving curriculum
episodes for replay is a separate reporting change, not implemented by this switch.

PR #217 archives a different historical experiment whose regular evaluation used
96-step routes. Its as-run YAML and historical reproduction stay intact. Future
curriculum-only variants should use this switch after the implementation reaches
the target branch; they must not be labeled byte-identical reproductions.

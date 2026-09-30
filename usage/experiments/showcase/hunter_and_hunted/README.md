# Hunter and hunted

This showcase demonstrates heterogeneous multi-agent behavior. Agents are
executed in YAML order, so `hunted` moves before `hunter` on each environment
step. The hunted uses normal cursor navigation, while the hunter selects the
same 26 directions but its Rust outcome moves two voxels at once.

Capture occurs when both cursors occupy the same or adjacent voxels. Capture
ends the episode and rewards the hunter. Reaching `max_steps` without capture
ends the episode and rewards the hunted. No other reward is configured.

```powershell
anysearch compile usage\experiments\showcase\hunter_and_hunted
anysearch run usage\experiments\showcase\hunter_and_hunted\experiment.yaml
```

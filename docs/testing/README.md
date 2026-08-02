# Testing

The automated pull-request workflow runs tests that do not start or require a
Ray runtime:

```powershell
python -m pytest -m "not ray and not integration" -q
```

This suite includes unit tests and other lightweight tests that may import or
mock Ray APIs. It excludes tests marked `ray`, which start or require a real Ray
runtime, and broader tests marked `integration`.

## Marker contract

- `ray`: starts or requires a real Ray runtime, including RLlib training and
  Tune sweeps.
- `integration`: exercises multiple runtime components or external processes.

Any test that calls `ray.init`, uses the `ray_session` fixture, launches an
RLlib algorithm, or executes a Tune sweep must carry the `ray` marker. A whole
module can declare it with:

```python
pytestmark = [pytest.mark.integration, pytest.mark.ray]
```

## Running excluded tests

Run the real-Ray suite explicitly on a machine with the required resources:

```powershell
python -m pytest -m ray -q
```

Run all integration tests, including non-Ray integrations, with:

```powershell
python -m pytest -m integration -q
```

The GitHub Actions workflow intentionally does not run either command.

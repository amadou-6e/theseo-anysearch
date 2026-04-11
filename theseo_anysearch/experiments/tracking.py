"""MLflow tracking helpers for experiments and hyperparameter sweeps."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from theseo_anysearch.experiments.models import ExperimentConfig, MLflowConfig

# Top-level keys excluded from param logging
_EXCLUDED_TOP_KEYS = {"anyscale", "renders", "mlflow"}
# Sub-keys excluded within specific sections
_EXCLUDED_TRAINING_KEYS = {"output_dir"}
_EXCLUDED_EXPERIMENT_KEYS = {"output_dir"}


def _flatten_dict(d: dict, prefix: str = "", result: dict | None = None) -> dict[str, Any]:
    """Recursively flatten a nested dict to dotted keys, skipping None values."""
    if result is None:
        result = {}
    for key, value in d.items():
        if value is None:
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_dict(value, full_key, result)
        elif isinstance(value, (list, tuple)):
            result[full_key] = json.dumps(value)
        else:
            result[full_key] = value
    return result


def _flatten_config(config: "ExperimentConfig") -> dict[str, Any]:
    """
    Flatten an ExperimentConfig to a dict with dotted keys suitable for
    mlflow.log_params().

    Excludes anyscale, renders, mlflow, and output_dir fields.
    Lists and dicts are JSON-encoded as strings (MLflow param value limit).
    """
    raw: dict[str, Any] = config.model_dump(by_alias=True, mode="json")
    for key in _EXCLUDED_TOP_KEYS:
        raw.pop(key, None)
    if "training" in raw:
        for k in _EXCLUDED_TRAINING_KEYS:
            raw["training"].pop(k, None)
    if "experiment" in raw:
        for k in _EXCLUDED_EXPERIMENT_KEYS:
            raw["experiment"].pop(k, None)
    return _flatten_dict(raw)


class MLflowTracker:
    """
    Thin, stateful wrapper around the mlflow client.

    All public methods are no-ops when mlflow_config is None, so callers can
    create a tracker unconditionally without guarding against None.

    Nesting (parent/child runs for tuning) delegates to MLflow's own run stack:
    start_child_run() opens a nested run; end_child_run() closes it.
    """

    def __init__(
        self,
        mlflow_config: "MLflowConfig | None",
        experiment_name: str,
    ) -> None:
        self._enabled = mlflow_config is not None
        self._run_id: str | None = None
        self._child_run_id: str | None = None
        self._tracking_uri: str | None = None
        if not self._enabled:
            return
        try:
            import mlflow
            if mlflow_config.tracking_uri:  # type: ignore[union-attr]
                mlflow.set_tracking_uri(mlflow_config.tracking_uri)  # type: ignore[union-attr]
                self._tracking_uri = mlflow_config.tracking_uri  # type: ignore[union-attr]
            else:
                self._tracking_uri = mlflow.get_tracking_uri()
            eff_name = (
                mlflow_config.experiment_name or experiment_name  # type: ignore[union-attr]
            )
            mlflow.set_experiment(eff_name)
        except Exception as exc:
            warnings.warn(f"MLflowTracker init failed: {exc}", stacklevel=2)
            self._enabled = False

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_name: str, tags: dict | None = None) -> str | None:
        """Open a new top-level MLflow run. Returns the mlflow run_id or None."""
        if not self._enabled:
            return None
        try:
            import mlflow
            active = mlflow.start_run(run_name=run_name, tags=tags)
            self._run_id = active.info.run_id
            return self._run_id
        except Exception as exc:
            warnings.warn(f"MLflow start_run failed: {exc}", stacklevel=2)
            return None

    def start_child_run(
        self,
        run_name: str,
        tags: dict | None = None,
        parent_run_id: str | None = None,
    ) -> str | None:
        """Open a nested child run (for tuning trials). Returns run_id or None.

        Pass ``parent_run_id`` when calling from a Ray worker process where the
        parent run is not active in the current process (cross-process nesting).
        When omitted, falls back to ``nested=True`` which requires an active
        parent run in the same process.
        """
        if not self._enabled:
            return None
        try:
            import mlflow
            if parent_run_id:
                active = mlflow.start_run(
                    run_name=run_name,
                    tags=tags,
                    parent_run_id=parent_run_id,
                )
            else:
                active = mlflow.start_run(run_name=run_name, tags=tags, nested=True)
            self._child_run_id = active.info.run_id
            return self._child_run_id
        except Exception as exc:
            warnings.warn(f"MLflow start_child_run failed: {exc}", stacklevel=2)
            return None

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active (outermost) run. status: FINISHED | FAILED | KILLED."""
        if not self._enabled:
            return
        try:
            import mlflow
            mlflow.end_run(status=status)
        except Exception as exc:
            warnings.warn(f"MLflow end_run failed: {exc}", stacklevel=2)

    def end_child_run(self, status: str = "FINISHED") -> None:
        """End the child run opened by start_child_run."""
        if not self._enabled:
            return
        if self._child_run_id:
            # Use MlflowClient directly so we target the exact run_id regardless
            # of the active-run stack (avoids cross-process stack confusion when
            # parent_run_id was used to open the child run).
            try:
                import mlflow
                mlflow.MlflowClient(tracking_uri=self._tracking_uri).set_terminated(
                    self._child_run_id, status
                )
                self._child_run_id = None
            except Exception as exc:
                warnings.warn(f"MLflow end_child_run failed: {exc}", stacklevel=2)
        else:
            self.end_run(status=status)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a flat dict of hyperparameters (called once at run start)."""
        if not self._enabled:
            return
        try:
            import mlflow
            # MLflow param values must be str; truncate to 500 chars (API limit)
            str_params = {k: str(v)[:500] for k, v in params.items()}
            mlflow.log_params(str_params)
        except Exception as exc:
            warnings.warn(f"MLflow log_params failed: {exc}", stacklevel=2)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of numeric metrics at the given training iteration."""
        if not self._enabled:
            return
        try:
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        except Exception as exc:
            warnings.warn(f"MLflow log_metrics failed: {exc}", stacklevel=2)

    def log_artifact(self, local_path: str | Path) -> None:
        """Upload a local file or directory to the active run's artifact store."""
        if not self._enabled:
            return
        try:
            import mlflow
            mlflow.log_artifact(str(local_path))
        except Exception as exc:
            warnings.warn(f"MLflow log_artifact failed: {exc}", stacklevel=2)

    def set_tag(self, key: str, value: str) -> None:
        """Set a string tag on the active run."""
        if not self._enabled:
            return
        try:
            import mlflow
            mlflow.set_tag(key, value)
        except Exception as exc:
            warnings.warn(f"MLflow set_tag failed: {exc}", stacklevel=2)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def run_url(self) -> str | None:
        """
        Tracking URL for the active run, or None when disabled / local store.
        Format: {tracking_uri}/#/experiments/{exp_id}/runs/{run_id}
        """
        if not self._enabled or self._run_id is None:
            return None
        try:
            if not self._tracking_uri or not self._tracking_uri.startswith("http"):
                return None
            import mlflow
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(self._run_id)
            exp_id = run.info.experiment_id
            return (
                f"{self._tracking_uri.rstrip('/')}/#/experiments/{exp_id}"
                f"/runs/{self._run_id}"
            )
        except Exception:
            return None

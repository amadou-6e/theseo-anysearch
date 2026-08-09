"""Streamlit application for interactive policy observation editing."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from theseo_anysearch.rllib.explain.scenarios import ObservationScenario
from theseo_anysearch.rllib.explain.service import resolve_run_dir
from theseo_anysearch.rllib.explain.ui.artifacts import build_artifact_bundle
from theseo_anysearch.rllib.explain.ui.editor import ObservationEditor
from theseo_anysearch.rllib.explain.ui.session import InteractiveExplanationSession


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ref", required=True)
    parser.add_argument("--checkpoint", default="latest")
    arguments, _ = parser.parse_known_args()
    return arguments


@st.cache_resource
def _session(run_ref: str, checkpoint: str) -> InteractiveExplanationSession:
    """Restore a checkpoint once for all UI reruns in this browser session."""

    return InteractiveExplanationSession(resolve_run_dir(run_ref), checkpoint)


def _load_uploaded(uploaded: object) -> dict[str, np.ndarray]:
    """Read an exact observation JSON or observation-scenario YAML upload."""

    raw = uploaded.getvalue().decode("utf-8")
    payload = yaml.safe_load(raw)
    if isinstance(payload, dict) and payload.get("type") == "observation":
        scenario = ObservationScenario.model_validate(payload)
        return {
            name: np.asarray(value, dtype=np.float32)
            for name, value in scenario.observation.items()
        }
    if not isinstance(payload, dict):
        raise ValueError("uploaded observation must contain a mapping")
    return {
        name: np.asarray(value, dtype=np.float32)
        for name, value in payload.items()
    }


def _edit_non_spatial(editor: ObservationEditor) -> None:
    """Render schema-driven controls for all non-spatial fields."""

    for name, value in editor.values.items():
        if name == "local_grid":
            continue
        low, high = editor.field_bounds(name)
        flat = value.reshape(-1)
        edited: list[float] = []
        st.sidebar.markdown(f"**{name}**")
        for index, current in enumerate(flat):
            lower = float(np.broadcast_to(low, value.shape).reshape(-1)[index])
            upper = float(np.broadcast_to(high, value.shape).reshape(-1)[index])
            edited.append(
                st.sidebar.number_input(
                    f"{name}[{index}]",
                    min_value=lower,
                    max_value=upper,
                    value=float(current),
                    key=f"field-{name}-{index}",
                )
            )
        editor.set_field(name, np.asarray(edited, dtype=np.float32).reshape(value.shape))


def main() -> None:
    """Render the interactive explainability interface."""

    args = _arguments()
    st.set_page_config(page_title="AnySearch policy explanation", layout="wide")
    st.title("AnySearch policy explanation")
    session = _session(args.run_ref, args.checkpoint)
    uploaded = st.sidebar.file_uploader(
        "Load observation or scenario", type=["json", "yaml", "yml"]
    )
    if "observation" not in st.session_state:
        st.session_state.observation = session.initial_observation()
    if uploaded is not None:
        upload_id = (uploaded.name, uploaded.size)
        if st.session_state.get("upload_id") != upload_id:
            st.session_state.observation = _load_uploaded(uploaded)
            st.session_state.upload_id = upload_id
            for key in tuple(st.session_state):
                if key.startswith("field-") or key.startswith("slice-"):
                    del st.session_state[key]

    editor = ObservationEditor(session.observation_space, st.session_state.observation)
    _edit_non_spatial(editor)

    if "local_grid" not in editor.values:
        raise ValueError("the initial explainability UI requires a box observation")
    axis = st.selectbox("Slice axis", ["x", "y", "z"])
    index = st.slider("Slice index", 0, editor.box_side - 1, editor.box_side // 2)
    st.caption("Voxel values are normalized network inputs in the range declared by the checkpoint.")
    frame = pd.DataFrame(editor.slice(axis, index))
    changed = st.data_editor(frame, use_container_width=True, key=f"slice-{axis}-{index}")
    editor.set_slice(axis, index, changed.to_numpy(dtype=np.float32))
    st.session_state.observation = editor.values

    report = session.explain(editor.values)
    step = report.steps[0]
    left, right = st.columns(2)
    left.metric("Selected action", f"{step.chosen_action} {step.chosen_direction}")
    right.metric("Q-value margin vs safe action", f"{step.score_margin:.6g}")
    st.subheader("Action scores")
    st.bar_chart(pd.DataFrame({"score": step.action_scores}))
    st.subheader("Grouped attribution")
    st.bar_chart(pd.DataFrame({"attribution": step.group_attributions}))
    st.json(report.to_json_dict())

    scenario_yaml = yaml.safe_dump(editor.to_scenario(), sort_keys=False)
    st.download_button("Download scenario YAML", scenario_yaml, "scenario.yaml")
    st.download_button(
        "Download report JSON",
        json.dumps(report.to_json_dict(), indent=2),
        "report.json",
    )
    st.download_button(
        "Download complete artifact bundle",
        build_artifact_bundle(report, editor.values, editor.to_scenario()),
        "explanation.zip",
        mime="application/zip",
    )


if __name__ == "__main__":
    main()

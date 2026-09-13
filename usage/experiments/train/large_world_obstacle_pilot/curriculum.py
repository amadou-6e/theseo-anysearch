"""Twelve exact-length gate routes, doubling from 2 to 4096 actions."""

from __future__ import annotations

from theseo_anysearch.rllib.trainer.waypoint_routes import WaypointRoute

STAGE_LENGTHS = tuple(1 << exponent for exponent in range(1, 13))
MAX_EPISODE_STEPS = 4608


def gate_routes(
    extent: tuple[int, int, int] = (4096, 2048, 512),
    portal_center: tuple[int, int] = (1024, 256),
) -> tuple[WaypointRoute, ...]:
    """Follow the portal axis; add a final turn to reach 4096 actions.

    The native occupancy query rejects the high X boundary, so the longest
    safe straight X traverse here is 4094 actions (X=1 through X=4095).
    Two Y actions after the last gate make the final route exactly 4096.
    """
    if extent[0] != STAGE_LENGTHS[-1]:
        raise ValueError("gate curriculum requires an X extent of 4096")
    if not (1 <= portal_center[0] + 2 < extent[1] and 1 <= portal_center[1] < extent[2]):
        raise ValueError("portal center must fit inside the task extent")
    start = (1, *portal_center)
    routes = []
    for length in STAGE_LENGTHS:
        if length < extent[0]:
            waypoints = ((start[0] + length, *portal_center),)
        else:
            waypoints = (
                (extent[0] - 1, *portal_center),
                (extent[0] - 1, portal_center[0] + 2, portal_center[1]),
            )
        routes.append(WaypointRoute(start=start, waypoints=waypoints))
    return tuple(routes)


def gate_curriculum_settings() -> dict:
    """Return the route schedule in the trainer's fixed-route config shape."""
    routes = gate_routes()
    return {
        "enabled": True,
        "completion_mode": "continue_route",
        "initial_start": routes[0].start,
        "initial_goal": routes[0].waypoints[0],
        "routes": [route.model_dump(mode="python") for route in routes],
        "advance": {"mode": "success", "successes_required": 1},
    }

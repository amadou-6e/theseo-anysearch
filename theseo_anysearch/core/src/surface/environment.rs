use std::collections::{HashMap, HashSet};

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::Serialize;

use crate::voxel::world::Coord;

use crate::environments::{Environment, StepResult};

#[derive(Clone, Debug, Serialize)]
pub struct AgentState {
    pub current: Coord,
    pub target: Coord,
    pub reached: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SurfaceObservation {
    pub steps_remaining: u32,
    pub reached_agents: usize,
    pub total_agents: usize,
    pub agents: Vec<AgentState>,
}

#[derive(Clone, Debug)]
pub enum SurfaceAction {
    StepAll,
}

pub struct SurfaceEnv {
    geometry: HashSet<Coord>,
    start_surface: Vec<Coord>,
    bounds_min: Coord,
    bounds_max: Coord,
    dims: (usize, usize, usize),
    exterior_mask: Vec<u8>, // 0 = unknown/interior, 1 = geometry, 2 = exterior
    agents: Vec<AgentState>,
    blocked: HashSet<Coord>,
    trail_owner: HashMap<Coord, usize>,
    max_steps: u32,
    steps: u32,
}

impl SurfaceEnv {
    pub fn from_surface(surface_coords: &[Coord], agent_count: usize, max_steps: u32) -> Self {
        let surface_set: HashSet<Coord> = surface_coords.iter().copied().collect();
        let (bounds_min, bounds_max) = compute_bounds(surface_coords, 0);
        let dims = dims_from_bounds(bounds_min, bounds_max);
        let exterior_mask = vec![2u8; dims.0 * dims.1 * dims.2];
        Self {
            geometry: HashSet::new(),
            start_surface: surface_set.iter().copied().collect(),
            bounds_min,
            bounds_max,
            dims,
            exterior_mask,
            agents: vec![
                AgentState {
                    current: (0, 0, 0),
                    target: (0, 0, 0),
                    reached: false,
                };
                agent_count
            ],
            blocked: HashSet::new(),
            trail_owner: HashMap::new(),
            max_steps,
            steps: 0,
        }
    }

    pub fn from_filled_surface(
        filled_coords: &[Coord],
        agent_count: usize,
        max_steps: u32,
    ) -> Self {
        let geometry: HashSet<Coord> = filled_coords.iter().copied().collect();
        let (bounds_min, bounds_max) = compute_bounds(filled_coords, 100);
        let dims = dims_from_bounds(bounds_min, bounds_max);
        let exterior_mask = build_exterior_mask(&geometry, bounds_min, bounds_max, dims);
        let start_surface =
            extract_exterior_shell(&geometry, bounds_min, bounds_max, dims, &exterior_mask);
        Self {
            geometry,
            start_surface,
            bounds_min,
            bounds_max,
            dims,
            exterior_mask,
            agents: vec![
                AgentState {
                    current: (0, 0, 0),
                    target: (0, 0, 0),
                    reached: false,
                };
                agent_count
            ],
            blocked: HashSet::new(),
            trail_owner: HashMap::new(),
            max_steps,
            steps: 0,
        }
    }

    pub fn surface_count(&self) -> usize {
        self.start_surface.len()
    }

    pub fn surface_coords(&self) -> Vec<Coord> {
        self.start_surface.clone()
    }

    pub fn trail_owner(&self) -> &HashMap<Coord, usize> {
        &self.trail_owner
    }

    fn is_exterior(&self, coord: Coord) -> bool {
        if let Some(idx) = coord_to_idx(coord, self.bounds_min, self.bounds_max, self.dims) {
            return self.exterior_mask[idx] == 2;
        }
        false
    }

    fn randomize_agents(&mut self, seed: u64) {
        self.blocked.clear();
        self.trail_owner.clear();

        let mut rng = StdRng::seed_from_u64(seed);
        let mut coords: Vec<Coord> = self.start_surface.clone();
        coords.shuffle(&mut rng);

        if coords.len() < 2 {
            return;
        }

        for (idx, agent) in self.agents.iter_mut().enumerate() {
            let start_index = idx * 2;
            let target_index = idx * 2 + 1;
            if target_index >= coords.len() {
                break;
            }
            agent.current = coords[start_index];
            agent.target = coords[target_index];
            agent.reached = agent.current == agent.target;

            self.blocked.insert(agent.current);
            self.trail_owner.entry(agent.current).or_insert(idx);
        }
    }

    fn as_observation(&self) -> SurfaceObservation {
        let reached_agents = self.agents.iter().filter(|a| a.reached).count();
        SurfaceObservation {
            steps_remaining: self.max_steps.saturating_sub(self.steps),
            reached_agents,
            total_agents: self.agents.len(),
            agents: self.agents.clone(),
        }
    }
}

impl Environment for SurfaceEnv {
    type Action = SurfaceAction;
    type Observation = SurfaceObservation;

    fn reset(&mut self, seed: u64) -> SurfaceObservation {
        self.steps = 0;
        self.randomize_agents(seed);
        self.as_observation()
    }

    fn step(&mut self, action: SurfaceAction) -> StepResult<SurfaceObservation> {
        self.steps += 1;
        let mut reward = 0.0;

        match action {
            SurfaceAction::StepAll => {
                let bounds_min = self.bounds_min;
                let bounds_max = self.bounds_max;
                let dims = self.dims;
                let mask = self.exterior_mask.clone();
                for (agent_idx, agent) in self.agents.iter_mut().enumerate() {
                    if agent.reached {
                        continue;
                    }

                    if let Some(next) = next_step_towards(
                        agent.current,
                        agent.target,
                        |c| {
                            coord_to_idx(c, bounds_min, bounds_max, dims)
                                .map(|i| mask[i] == 2)
                                .unwrap_or(false)
                        },
                        &self.blocked,
                    ) {
                        agent.current = next;
                        reward += 0.02;
                        if !self.blocked.contains(&next) {
                            self.blocked.insert(next);
                            self.trail_owner.insert(next, agent_idx);
                        }
                    } else {
                        reward -= 0.05;
                    }

                    if agent.current == agent.target {
                        agent.reached = true;
                        reward += 1.0;
                    }
                }
            }
        }

        let observation = self.as_observation();
        let done =
            self.steps >= self.max_steps || observation.reached_agents == observation.total_agents;
        StepResult {
            observation,
            reward,
            done,
        }
    }
}

fn neighbors(coord: Coord) -> [Coord; 6] {
    let (x, y, z) = coord;
    [
        (x.saturating_add(1), y, z),
        (x.saturating_sub(1), y, z),
        (x, y.saturating_add(1), z),
        (x, y.saturating_sub(1), z),
        (x, y, z.saturating_add(1)),
        (x, y, z.saturating_sub(1)),
    ]
}

fn extract_exterior_shell(
    filled: &HashSet<Coord>,
    bounds_min: Coord,
    bounds_max: Coord,
    dims: (usize, usize, usize),
    mask: &[u8],
) -> Vec<Coord> {
    let mut shell = HashSet::new();
    for coord in filled {
        for n in neighbors(*coord) {
            if !filled.contains(&n)
                && coord_to_idx(n, bounds_min, bounds_max, dims)
                    .map(|i| mask[i] == 2)
                    .unwrap_or(false)
            {
                shell.insert(n);
            }
        }
    }
    shell.into_iter().collect()
}

fn compute_bounds(coords: &[Coord], margin: u16) -> (Coord, Coord) {
    if coords.is_empty() {
        return ((0, 0, 0), (0, 0, 0));
    }
    let min_x = coords
        .iter()
        .map(|c| c.0)
        .min()
        .unwrap_or(0)
        .saturating_sub(margin);
    let min_y = coords
        .iter()
        .map(|c| c.1)
        .min()
        .unwrap_or(0)
        .saturating_sub(margin);
    let min_z = coords
        .iter()
        .map(|c| c.2)
        .min()
        .unwrap_or(0)
        .saturating_sub(margin);
    let max_x = coords
        .iter()
        .map(|c| c.0)
        .max()
        .unwrap_or(0)
        .saturating_add(margin);
    let max_y = coords
        .iter()
        .map(|c| c.1)
        .max()
        .unwrap_or(0)
        .saturating_add(margin);
    let max_z = coords
        .iter()
        .map(|c| c.2)
        .max()
        .unwrap_or(0)
        .saturating_add(margin);
    ((min_x, min_y, min_z), (max_x, max_y, max_z))
}

fn dims_from_bounds(min: Coord, max: Coord) -> (usize, usize, usize) {
    (
        (max.0 - min.0 + 1) as usize,
        (max.1 - min.1 + 1) as usize,
        (max.2 - min.2 + 1) as usize,
    )
}

fn coord_to_idx(
    coord: Coord,
    min: Coord,
    max: Coord,
    dims: (usize, usize, usize),
) -> Option<usize> {
    if coord.0 < min.0
        || coord.0 > max.0
        || coord.1 < min.1
        || coord.1 > max.1
        || coord.2 < min.2
        || coord.2 > max.2
    {
        return None;
    }
    let x = (coord.0 - min.0) as usize;
    let y = (coord.1 - min.1) as usize;
    let z = (coord.2 - min.2) as usize;
    Some(x + y * dims.0 + z * dims.0 * dims.1)
}

fn build_exterior_mask(
    geometry: &HashSet<Coord>,
    bounds_min: Coord,
    bounds_max: Coord,
    dims: (usize, usize, usize),
) -> Vec<u8> {
    let mut mask = vec![0u8; dims.0 * dims.1 * dims.2];
    for &g in geometry {
        if let Some(i) = coord_to_idx(g, bounds_min, bounds_max, dims) {
            mask[i] = 1;
        }
    }

    let mut queue = std::collections::VecDeque::new();
    for z in 0..dims.2 {
        for y in 0..dims.1 {
            for x in 0..dims.0 {
                if x != 0
                    && x != dims.0 - 1
                    && y != 0
                    && y != dims.1 - 1
                    && z != 0
                    && z != dims.2 - 1
                {
                    continue;
                }
                let idx = x + y * dims.0 + z * dims.0 * dims.1;
                if mask[idx] == 0 {
                    mask[idx] = 2;
                    queue.push_back((x, y, z));
                }
            }
        }
    }

    while let Some((x, y, z)) = queue.pop_front() {
        let neigh = [
            (x.wrapping_add(1), y, z, x + 1 < dims.0),
            (x.wrapping_sub(1), y, z, x > 0),
            (x, y.wrapping_add(1), z, y + 1 < dims.1),
            (x, y.wrapping_sub(1), z, y > 0),
            (x, y, z.wrapping_add(1), z + 1 < dims.2),
            (x, y, z.wrapping_sub(1), z > 0),
        ];
        for (nx, ny, nz, ok) in neigh {
            if !ok {
                continue;
            }
            let ni = nx + ny * dims.0 + nz * dims.0 * dims.1;
            if mask[ni] == 0 {
                mask[ni] = 2;
                queue.push_back((nx, ny, nz));
            }
        }
    }

    mask
}

fn next_step_towards(
    start: Coord,
    goal: Coord,
    mut is_exterior: impl FnMut(Coord) -> bool,
    blocked: &HashSet<Coord>,
) -> Option<Coord> {
    if start == goal {
        return Some(start);
    }
    let mut candidates = neighbors(start)
        .into_iter()
        .filter(|n| is_exterior(*n))
        .filter(|n| !blocked.contains(n) || *n == goal)
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by_key(|c| manhattan(*c, goal));
    Some(candidates[0])
}

fn manhattan(a: Coord, b: Coord) -> u32 {
    (a.0.abs_diff(b.0) as u32) + (a.1.abs_diff(b.1) as u32) + (a.2.abs_diff(b.2) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 20-coord straight line: easy for one agent to traverse.
    fn line_surface() -> Vec<Coord> {
        (0u16..20).map(|i| (100 + i, 100, 100)).collect()
    }

    /// 10×10×10 solid cube whose exterior shell forms the surface.
    fn filled_cube() -> Vec<Coord> {
        let mut coords = Vec::new();
        for x in 100u16..110 {
            for y in 100u16..110 {
                for z in 100u16..110 {
                    coords.push((x, y, z));
                }
            }
        }
        coords
    }

    // ── from_filled_surface ──────────────────────────────────────────────────

    #[test]
    fn from_filled_surface_builds_exterior() {
        let cube = filled_cube();
        let geometry: HashSet<Coord> = cube.iter().copied().collect();
        let env = SurfaceEnv::from_filled_surface(&cube, 1, 50);
        // Surface shell must be non-empty and lie outside the solid geometry
        assert!(env.surface_count() > 0);
        for c in env.surface_coords() {
            assert!(!geometry.contains(&c), "surface cell is inside geometry");
        }
    }

    // ── reset ────────────────────────────────────────────────────────────────

    #[test]
    fn reset_assigns_start_goal() {
        let surface = line_surface();
        let surface_set: HashSet<Coord> = surface.iter().copied().collect();
        let mut env = SurfaceEnv::from_surface(&surface, 2, 50);
        let obs = env.reset(42);
        assert_eq!(obs.total_agents, 2);
        for agent in &obs.agents {
            assert!(surface_set.contains(&agent.current));
            assert!(surface_set.contains(&agent.target));
        }
    }

    #[test]
    fn deterministic_same_seed() {
        // Resetting the same env twice with the same seed must yield identical assignments.
        let surface = line_surface();
        let mut env = SurfaceEnv::from_surface(&surface, 2, 50);
        let obs1 = env.reset(42);
        let snap1: Vec<_> = obs1.agents.iter().map(|a| (a.current, a.target)).collect();
        let obs2 = env.reset(42);
        let snap2: Vec<_> = obs2.agents.iter().map(|a| (a.current, a.target)).collect();
        assert_eq!(snap1, snap2);
    }

    #[test]
    fn different_seeds_differ() {
        let surface = line_surface();
        let mut env1 = SurfaceEnv::from_surface(&surface, 2, 50);
        let mut env2 = SurfaceEnv::from_surface(&surface, 2, 50);
        let obs1 = env1.reset(1);
        let obs2 = env2.reset(2);
        // Different seeds should (almost certainly) produce different assignments
        let same = obs1.agents[0].current == obs2.agents[0].current
            && obs1.agents[0].target == obs2.agents[0].target;
        assert!(!same, "different seeds produced identical assignments");
    }

    // ── step ─────────────────────────────────────────────────────────────────

    #[test]
    fn step_all_advances_agents() {
        let surface = line_surface();
        let mut env = SurfaceEnv::from_surface(&surface, 1, 50);
        let obs0 = env.reset(42);
        let start = obs0.agents[0].current;
        let sr = env.step(SurfaceAction::StepAll);
        let end = sr.observation.agents[0].current;
        // Agent either moved or immediately reached its goal
        assert!(end != start || sr.observation.agents[0].reached);
    }

    #[test]
    fn reached_agent_counted() {
        let surface = line_surface();
        let mut env = SurfaceEnv::from_surface(&surface, 1, 200);
        env.reset(42);
        let mut last_reached = 0;
        loop {
            let sr = env.step(SurfaceAction::StepAll);
            last_reached = sr.observation.reached_agents;
            if sr.done {
                break;
            }
        }
        assert_eq!(last_reached, 1);
    }

    #[test]
    fn trail_blocks_cells() {
        let surface = line_surface();
        let mut env = SurfaceEnv::from_surface(&surface, 1, 50);
        env.reset(42);
        // trail_owner already has the starting cell after reset
        assert!(!env.trail_owner().is_empty());
        env.step(SurfaceAction::StepAll);
        // After a step, trail must have grown (or stayed if agent immediately reached goal)
        assert!(!env.trail_owner().is_empty());
    }

    #[test]
    fn done_when_all_reached() {
        let surface = line_surface();
        let mut env = SurfaceEnv::from_surface(&surface, 1, 200);
        env.reset(42);
        let mut done = false;
        for _ in 0..200 {
            let sr = env.step(SurfaceAction::StepAll);
            if sr.done {
                done = true;
                assert!(sr.observation.reached_agents > 0 || sr.observation.steps_remaining == 0);
                break;
            }
        }
        assert!(done);
    }

    #[test]
    fn done_at_max_steps() {
        let surface = line_surface();
        let mut env = SurfaceEnv::from_surface(&surface, 1, 3);
        env.reset(42);
        env.step(SurfaceAction::StepAll);
        env.step(SurfaceAction::StepAll);
        let sr = env.step(SurfaceAction::StepAll);
        assert!(sr.done);
    }

    #[test]
    fn reward_success() {
        let surface = line_surface();
        let mut env = SurfaceEnv::from_surface(&surface, 1, 200);
        env.reset(42);
        let mut got_success = false;
        for _ in 0..200 {
            let sr = env.step(SurfaceAction::StepAll);
            if sr.reward >= 1.0 {
                got_success = true;
            }
            if sr.done {
                break;
            }
        }
        assert!(got_success, "agent never reached its goal");
    }
}

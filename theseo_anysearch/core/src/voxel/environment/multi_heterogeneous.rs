//! Ordered heterogeneous-agent stepping and asymmetric capture tasks.

use std::path::Path;

use serde::Deserialize;

use super::{
    geometry::{l2, manhattan},
    multi::{MultiAgentVoxelEnv, MultiStepResult},
    multi_action::{AgentActionPipeline, AgentPipelineSpec},
    multi_action_execute::execute_agent_action,
};

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct CaptureTask {
    pub hunter: String,
    pub hunted: String,
    pub capture_distance: u16,
    pub hunter_capture_reward: f32,
    pub hunted_escape_reward: f32,
}

impl MultiAgentVoxelEnv {
    pub fn configure_agents(
        &mut self,
        agents_json: &str,
        native_library: Option<&Path>,
    ) -> Result<(), String> {
        let specs: Vec<AgentPipelineSpec> = serde_json::from_str(agents_json)
            .map_err(|error| format!("invalid heterogeneous agents: {error}"))?;
        if specs.len() != self.agent_count {
            return Err(format!(
                "configured {} agents but agent_count is {}",
                specs.len(),
                self.agent_count
            ));
        }
        let pipelines = specs
            .into_iter()
            .map(|spec| AgentActionPipeline::load(spec, native_library))
            .collect::<Result<Vec<_>, _>>()?;
        self.pipelines = pipelines;
        Ok(())
    }

    pub fn configure_capture_task(&mut self, task_json: Option<&str>) -> Result<(), String> {
        self.capture_task = task_json
            .map(serde_json::from_str)
            .transpose()
            .map_err(|error| format!("invalid hunter_and_hunted task: {error}"))?;
        if let Some(task) = &self.capture_task {
            let ids: Vec<&str> = self
                .pipelines
                .iter()
                .map(|agent| agent.id.as_str())
                .collect();
            if !ids.contains(&task.hunter.as_str()) || !ids.contains(&task.hunted.as_str()) {
                return Err("capture task agents must exist in env.agents".to_owned());
            }
        }
        Ok(())
    }

    /// Apply actions in configured agent order, making earlier outcomes visible
    /// to every later agent in the same environment step.
    pub fn step_all(&mut self, actions: &[i32]) -> MultiStepResult {
        self.steps += 1;
        let mut rewards = vec![0.0; self.agent_count];
        let observation_filled = self.voxel_count();

        for index in 0..self.agent_count {
            if self.agents[index].goal_reached {
                continue;
            }
            let action = actions.get(index).copied().unwrap_or(-1);
            let previous = self.agents[index].cursor;
            let goal = self.agents[index].goal;
            let result = execute_agent_action(
                &mut self.pipelines[index],
                &mut self.world,
                action,
                previous,
                goal,
                self.steps,
                self.max_steps,
                self.grid_size,
                observation_filled,
            );
            let mut reward = 0.0;
            match result {
                Ok(action_result) => {
                    self.agents[index].cursor = action_result.cursor;
                    if action_result.collision {
                        reward += self.reward_config.collision_cost;
                    }
                }
                Err(error) => {
                    self.last_action_error = Some(error);
                    reward += self.reward_config.collision_cost;
                }
            }
            if let Some(goal) = goal {
                let distance = l2(self.agents[index].cursor, goal);
                reward += self.reward_config.base_step_reward(
                    self.agents[index].prev_l2,
                    distance,
                    self.grid_size,
                );
                self.agents[index].prev_l2 = distance;
                if self.agents[index].cursor == goal {
                    reward += self.reward_config.goal_reward;
                    self.agents[index].goal_reached = true;
                }
            }
            rewards[index] = reward;
            if self.apply_capture_rewards(&mut rewards) {
                return self.result(rewards, true);
            }
        }

        let any_have_goal = self.agents.iter().any(|agent| agent.goal.is_some());
        let goals_done = any_have_goal
            && self
                .agents
                .iter()
                .all(|agent| agent.goal.is_none() || agent.goal_reached);
        let timed_out = self.steps >= self.max_steps;
        if timed_out {
            self.apply_escape_reward(&mut rewards);
        }
        self.result(rewards, goals_done || timed_out)
    }

    fn apply_capture_rewards(&self, rewards: &mut [f32]) -> bool {
        let Some(task) = &self.capture_task else {
            return false;
        };
        let hunter = self.agent_index(&task.hunter).expect("validated hunter");
        let hunted = self.agent_index(&task.hunted).expect("validated hunted");
        let a = self.agents[hunter].cursor;
        let b = self.agents[hunted].cursor;
        if a.0
            .abs_diff(b.0)
            .max(a.1.abs_diff(b.1))
            .max(a.2.abs_diff(b.2))
            <= task.capture_distance
        {
            rewards[hunter] += task.hunter_capture_reward;
            return true;
        }
        false
    }

    fn apply_escape_reward(&self, rewards: &mut [f32]) {
        let Some(task) = &self.capture_task else {
            return;
        };
        let hunted = self.agent_index(&task.hunted).expect("validated hunted");
        rewards[hunted] += task.hunted_escape_reward;
    }

    fn agent_index(&self, id: &str) -> Option<usize> {
        self.pipelines.iter().position(|agent| agent.id == id)
    }

    fn result(&self, rewards: Vec<f32>, done: bool) -> MultiStepResult {
        MultiStepResult {
            steps_remaining: self.max_steps.saturating_sub(self.steps),
            voxel_count: self.voxel_count(),
            cursors: self.agents.iter().map(|agent| agent.cursor).collect(),
            goal_distances: self
                .agents
                .iter()
                .map(|agent| agent.goal.map(|goal| manhattan(agent.cursor, goal)))
                .collect(),
            rewards,
            done,
        }
    }
}

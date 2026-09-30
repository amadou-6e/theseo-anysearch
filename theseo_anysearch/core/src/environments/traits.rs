pub struct StepResult<O> {
    pub observation: O,
    pub reward: f32,
    pub done: bool,
}

pub trait Environment {
    type Action;
    type Observation;

    fn reset(&mut self, seed: u64) -> Self::Observation;
    fn step(&mut self, action: Self::Action) -> StepResult<Self::Observation>;
}

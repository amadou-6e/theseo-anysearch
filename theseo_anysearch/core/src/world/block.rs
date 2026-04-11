pub const BLOCK_KIND_FILLED: u8 = 1;
pub const BLOCK_KIND_START: u8 = 2;
pub const BLOCK_KIND_GOAL: u8 = 3;

#[derive(Clone, Debug)]
pub struct Block {
    pub kind: u8,
    pub active: bool,
    pub reward_weight: f32,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            kind: BLOCK_KIND_FILLED,
            active: true,
            reward_weight: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BlockUpdate {
    pub kind: Option<u8>,
    pub active: Option<bool>,
    pub reward_weight: Option<f32>,
}

impl BlockUpdate {
    pub fn apply_to(self, block: &mut Block) {
        if let Some(kind) = self.kind {
            block.kind = kind;
        }
        if let Some(active) = self.active {
            block.active = active;
        }
        if let Some(reward_weight) = self.reward_weight {
            block.reward_weight = reward_weight;
        }
    }
}

use crate::world::Coord;

#[derive(Clone, Debug)]
pub struct GeometrySubmission {
    pub name: String,
    pub stl_ascii: String,
    pub origin: Coord,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct EnvConfig {
    pub max_steps: u32,
}

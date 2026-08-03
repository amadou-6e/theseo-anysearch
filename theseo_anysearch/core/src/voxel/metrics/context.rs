#[derive(Clone, Debug)]
pub struct MetricContext {
    pub json: Vec<u8>,
}

impl MetricContext {
    pub fn new(json: impl Into<Vec<u8>>) -> Self {
        Self { json: json.into() }
    }
}

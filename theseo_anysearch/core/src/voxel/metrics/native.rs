use std::path::Path;

use libloading::{Library, Symbol};

use crate::voxel::common::load_library;

use super::{
    abi::{MetricFunctionV1, METRIC_BUFFER_SIZE},
    context::MetricContext,
    result::MetricResult,
};

#[derive(Clone, Copy, Debug)]
pub enum MetricScope {
    Training,
    Evaluation,
}

impl MetricScope {
    fn symbol(self) -> &'static [u8] {
        match self {
            Self::Training => b"anysearch_compute_training_metrics_v1\0",
            Self::Evaluation => b"anysearch_compute_evaluation_metrics_v1\0",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Training => "training",
            Self::Evaluation => "evaluation",
        }
    }
}

pub struct NativeMetricExtension {
    _library: Library,
    function: MetricFunctionV1,
    scope: MetricScope,
}

impl NativeMetricExtension {
    pub fn load(path: &Path, scope: MetricScope) -> Result<Self, String> {
        let library = load_library(path, "metric")?;
        let function = unsafe {
            let symbol: Symbol<MetricFunctionV1> = library
                .get(scope.symbol())
                .map_err(|error| format!("{} metrics are not exported: {error}", scope.name()))?;
            *symbol
        };
        Ok(Self {
            _library: library,
            function,
            scope,
        })
    }

    pub fn compute(&self, context: &MetricContext) -> Result<MetricResult, String> {
        let mut output = vec![0_u8; METRIC_BUFFER_SIZE];
        let mut length = 0_usize;
        let status = unsafe {
            (self.function)(
                context.json.as_ptr(),
                context.json.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut length,
            )
        };
        if status != 0 || length >= output.len() {
            return Err(format!(
                "native {} metrics failed with status {status}",
                self.scope.name()
            ));
        }
        serde_json::from_slice(&output[..length]).map_err(|error| {
            format!(
                "native {} metrics returned invalid JSON: {error}",
                self.scope.name()
            )
        })
    }
}

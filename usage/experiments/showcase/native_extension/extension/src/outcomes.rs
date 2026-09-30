use anysearch_extension::{anysearch_outcome, OutcomeContext, OutcomeMutations, OutcomeResult};

#[anysearch_outcome]
pub fn mark_destination(
    context: &OutcomeContext,
    mutations: &mut OutcomeMutations,
) -> OutcomeResult {
    mutations.place_voxel(context.destination);
    OutcomeResult::applied()
}

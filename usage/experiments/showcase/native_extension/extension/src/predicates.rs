use anysearch_extension::{anysearch_predicate, PredicateContext, PredicateResult};

#[anysearch_predicate]
pub fn avoid_repeated_collision(context: &PredicateContext) -> PredicateResult {
    let blocked = context
        .history
        .last()
        .is_some_and(|entry| entry.action_index == context.action_index && entry.collision);
    if blocked {
        PredicateResult::deny()
    } else {
        PredicateResult::allow()
    }
}

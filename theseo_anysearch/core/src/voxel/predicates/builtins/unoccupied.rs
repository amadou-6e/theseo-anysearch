pub fn evaluate(action_index: i32, destination_blocked: bool) -> bool {
    action_index == 26 || !destination_blocked
}

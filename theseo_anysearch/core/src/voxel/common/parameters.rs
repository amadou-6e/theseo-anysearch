pub fn validate_parameters(kind: &str, parameters_json: &str) -> Result<(), String> {
    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(parameters_json)
        .map(|_| ())
        .map_err(|error| format!("invalid {kind} parameters: {error}"))
}

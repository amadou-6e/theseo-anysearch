mod outcomes;
mod predicates;

const ABI_VERSION: u32 = 2;
const PREDICATE: u64 = 8;
const OUTCOME: u64 = 16;

#[no_mangle]
pub extern "C" fn anysearch_extension_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn anysearch_extension_capabilities() -> u64 {
    PREDICATE | OUTCOME
}
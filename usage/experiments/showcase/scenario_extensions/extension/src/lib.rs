mod scenarios;

const ABI_VERSION: u32 = 2;
const SCENARIO: u64 = 32;

#[no_mangle]
pub extern "C" fn anysearch_extension_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn anysearch_extension_capabilities() -> u64 {
    SCENARIO
}

mod rewards;

const ABI_VERSION: u32 = 2;
const REWARD: u64 = 1;

#[no_mangle]
pub extern "C" fn anysearch_extension_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn anysearch_extension_capabilities() -> u64 {
    REWARD
}

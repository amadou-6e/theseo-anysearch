use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn anysearch_reward(arguments: TokenStream, item: TokenStream) -> TokenStream {
    if !arguments.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "anysearch_reward does not accept arguments",
        )
        .to_compile_error()
        .into();
    }

    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_reward_{}_v2", function_name);

    quote! {
        #function

        #[doc(hidden)]
        #[no_mangle]
        pub unsafe extern "C" fn #export_name(
            context: *const ::anysearch_extension::RewardContextV2,
            result: *mut ::anysearch_extension::RewardResultV2,
        ) -> i32 {
            ::anysearch_extension::export_reward_v2(context, result, #function_name)
        }
    }
    .into()
}

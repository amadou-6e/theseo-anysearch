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
#[proc_macro_attribute]
pub fn anysearch_predicate(arguments: TokenStream, item: TokenStream) -> TokenStream {
    if !arguments.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "anysearch_predicate does not accept arguments",
        )
        .to_compile_error()
        .into();
    }
    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_predicate_{}_v2", function_name);
    quote! {
        #function
        #[doc(hidden)]
        #[no_mangle]
        pub unsafe extern "C" fn #export_name(
            context: *const ::anysearch_extension::PredicateContextV2,
            result: *mut ::anysearch_extension::PredicateResultV2,
        ) -> i32 {
            ::anysearch_extension::export_predicate_v2(context, result, #function_name)
        }
    }
    .into()
}

#[proc_macro_attribute]
pub fn anysearch_outcome(arguments: TokenStream, item: TokenStream) -> TokenStream {
    if !arguments.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "anysearch_outcome does not accept arguments",
        )
        .to_compile_error()
        .into();
    }
    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_outcome_{}_v2", function_name);
    quote! {
        #function
        #[doc(hidden)]
        #[no_mangle]
        pub unsafe extern "C" fn #export_name(
            context: *const ::anysearch_extension::OutcomeContextV2,
            result: *mut ::anysearch_extension::OutcomeResultV2,
        ) -> i32 {
            ::anysearch_extension::export_outcome_v2(context, result, #function_name)
        }
    }
    .into()
}

#[proc_macro_attribute]
pub fn anysearch_scenario(arguments: TokenStream, item: TokenStream) -> TokenStream {
    if !arguments.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "anysearch_scenario does not accept arguments",
        )
        .to_compile_error()
        .into();
    }
    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_scenario_{}_v1", function_name);
    quote! {
        #function
        #[doc(hidden)]
        #[no_mangle]
        pub unsafe extern "C" fn #export_name(
            input: *const u8,
            input_len: usize,
            output: *mut u8,
            output_capacity: usize,
            output_len: *mut usize,
        ) -> i32 {
            ::anysearch_extension::export_scenario_v1(
                input, input_len, output, output_capacity, output_len, #function_name
            )
        }
    }
    .into()
}

#[proc_macro_attribute]
pub fn anysearch_scenario_v2(arguments: TokenStream, item: TokenStream) -> TokenStream {
    if !arguments.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "anysearch_scenario_v2 does not accept arguments",
        )
        .to_compile_error()
        .into();
    }
    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_scenario_{}_v2", function_name);
    quote! {
        #function
        #[doc(hidden)]
        #[no_mangle]
        pub unsafe extern "C" fn #export_name(
            context: *const ::anysearch_extension::ScenarioContextV2Raw,
            output: *mut u8,
            output_capacity: usize,
            required_length: *mut usize,
        ) -> ::anysearch_extension::ScenarioStatusV2 {
            ::anysearch_extension::export_scenario_v2(context, output, output_capacity, required_length, #function_name)
        }
    }.into()
}

#[proc_macro_attribute]
pub fn anysearch_geometry_v1(arguments: TokenStream, item: TokenStream) -> TokenStream {
    if !arguments.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "anysearch_geometry_v1 does not accept arguments",
        )
        .to_compile_error()
        .into();
    }
    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_geometry_{}_v1", function_name);
    quote! {
        #function
        #[doc(hidden)]
        #[no_mangle]
        pub unsafe extern "C" fn #export_name(
            context: *const ::anysearch_extension::GeometryContextV1Raw,
            output: *mut u8,
            output_capacity: usize,
            required_length: *mut usize,
        ) -> ::anysearch_extension::GeometryStatusV1 {
            ::anysearch_extension::export_geometry_v1(
                context, output, output_capacity, required_length, #function_name
            )
        }
    }.into()
}

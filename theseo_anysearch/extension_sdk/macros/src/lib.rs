use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::Parser, parse_macro_input, punctuated::Punctuated, Expr, ExprLit, ItemFn, Lit,
    LitStr, Meta, Token,
};

#[derive(Default)]
struct MetadataOptions {
    version: Option<u32>,
    environment_families: Vec<String>,
    dependencies: Vec<(String, String)>,
    conflicts: Vec<(String, String)>,
}

fn parse_references(value: &LitStr) -> syn::Result<Vec<(String, String)>> {
    value
        .value()
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .map(|item| {
            let (kind, name) = item.trim().split_once(':').ok_or_else(|| {
                syn::Error::new(value.span(), "rule references must use kind:name")
            })?;
            rule_kind_variant(kind, value)?;
            if name.is_empty() {
                return Err(syn::Error::new(value.span(), "rule reference name is empty"));
            }
            Ok((kind.to_owned(), name.to_owned()))
        })
        .collect()
}

fn rule_kind_variant(kind: &str, source: &LitStr) -> syn::Result<syn::Ident> {
    let variant = match kind {
        "predicate" => "Predicate",
        "outcome" => "Outcome",
        "reward" => "Reward",
        "training_metrics" => "TrainingMetrics",
        "evaluation_metrics" => "EvaluationMetrics",
        "scenario" => "Scenario",
        _ => {
            return Err(syn::Error::new(
                source.span(),
                format!("unknown environment rule kind {kind:?}"),
            ))
        }
    };
    Ok(format_ident!("{variant}"))
}

fn parse_metadata_options(arguments: TokenStream) -> syn::Result<MetadataOptions> {
    let parser = Punctuated::<Meta, Token![,]>::parse_terminated;
    let metadata = parser.parse(arguments)?;
    let mut options = MetadataOptions::default();
    for item in metadata {
        let Meta::NameValue(item) = item else {
            return Err(syn::Error::new_spanned(item, "expected key = value"));
        };
        let Some(key) = item.path.get_ident().map(ToString::to_string) else {
            return Err(syn::Error::new_spanned(item.path, "expected metadata key"));
        };
        let Expr::Lit(ExprLit { lit, .. }) = item.value else {
            return Err(syn::Error::new_spanned(item.value, "expected a literal value"));
        };
        match (key.as_str(), lit) {
            ("version", Lit::Int(value)) => {
                let version = value.base10_parse::<u32>()?;
                if version == 0 {
                    return Err(syn::Error::new(value.span(), "version must be positive"));
                }
                options.version = Some(version);
            }
            ("environment_families", Lit::Str(value)) => {
                options.environment_families = value
                    .value()
                    .split(',')
                    .map(str::trim)
                    .filter(|family| !family.is_empty())
                    .map(|family| {
                        if !matches!(family, "voxel" | "surface") {
                            Err(syn::Error::new(
                                value.span(),
                                format!("unknown environment family {family:?}"),
                            ))
                        } else {
                            Ok(family.to_owned())
                        }
                    })
                    .collect::<syn::Result<Vec<_>>>()?;
                if options.environment_families.is_empty() {
                    return Err(syn::Error::new(
                        value.span(),
                        "environment_families cannot be empty",
                    ));
                }
            }
            ("dependencies", Lit::Str(value)) => {
                options.dependencies = parse_references(&value)?;
            }
            ("conflicts", Lit::Str(value)) => {
                options.conflicts = parse_references(&value)?;
            }
            (known, value) if matches!(known, "version" | "environment_families" | "dependencies" | "conflicts") => {
                return Err(syn::Error::new_spanned(value, "metadata value has the wrong type"));
            }
            (unknown, value) => {
                return Err(syn::Error::new_spanned(
                    value,
                    format!("unknown rule metadata key {unknown:?}"),
                ));
            }
        }
    }
    Ok(options)
}

fn metadata_export(
    function_name: &syn::Ident,
    kind: &str,
    options: &MetadataOptions,
) -> syn::Result<proc_macro2::TokenStream> {
    let kind_source = LitStr::new(kind, function_name.span());
    let kind_variant = rule_kind_variant(kind, &kind_source)?;
    let export_name = format_ident!("anysearch_rule_metadata_{}_{}_v1", kind, function_name);
    let name = function_name.to_string();
    let version = options.version.unwrap_or(1);
    let families = if options.environment_families.is_empty() {
        vec!["voxel".to_owned()]
    } else {
        options.environment_families.clone()
    };
    let family_literals = families
        .iter()
        .map(|value| LitStr::new(value, function_name.span()));
    let dependency_tokens = options
        .dependencies
        .iter()
        .map(|(kind, name)| {
            let source = LitStr::new(kind, function_name.span());
            let variant = rule_kind_variant(kind, &source)?;
            let name = LitStr::new(name, function_name.span());
            Ok(quote! { (#name, ::anysearch_extension::RuleKind::#variant) })
        })
        .collect::<syn::Result<Vec<_>>>()?;
    let conflict_tokens = options
        .conflicts
        .iter()
        .map(|(kind, name)| {
            let source = LitStr::new(kind, function_name.span());
            let variant = rule_kind_variant(kind, &source)?;
            let name = LitStr::new(name, function_name.span());
            Ok(quote! { (#name, ::anysearch_extension::RuleKind::#variant) })
        })
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        #[doc(hidden)]
        #[no_mangle]
        pub unsafe extern "C" fn #export_name(
            output: *mut u8,
            output_capacity: usize,
            required_length: *mut usize,
        ) -> i32 {
            let metadata = ::anysearch_extension::EnvironmentRuleMetadata::new(
                #name,
                ::anysearch_extension::RuleKind::#kind_variant,
            )
            .with_version(#version)
            .with_environment_families(&[#(#family_literals),*])
            .with_dependencies(&[#(#dependency_tokens),*])
            .with_conflicts(&[#(#conflict_tokens),*]);
            ::anysearch_extension::export_rule_metadata_v1(
                output,
                output_capacity,
                required_length,
                &metadata,
            )
        }
    })
}

#[proc_macro_attribute]
pub fn anysearch_reward(arguments: TokenStream, item: TokenStream) -> TokenStream {
    let options = match parse_metadata_options(arguments) {
        Ok(options) => options,
        Err(error) => return error.to_compile_error().into(),
    };

    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_reward_{}_v2", function_name);
    let metadata = metadata_export(function_name, "reward", &options)
        .unwrap_or_else(syn::Error::into_compile_error);

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
        #metadata
    }
    .into()
}
#[proc_macro_attribute]
pub fn anysearch_predicate(arguments: TokenStream, item: TokenStream) -> TokenStream {
    let options = match parse_metadata_options(arguments) {
        Ok(options) => options,
        Err(error) => return error.to_compile_error().into(),
    };
    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_predicate_{}_v2", function_name);
    let metadata = metadata_export(function_name, "predicate", &options)
        .unwrap_or_else(syn::Error::into_compile_error);
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
        #metadata
    }
    .into()
}

#[proc_macro_attribute]
pub fn anysearch_outcome(arguments: TokenStream, item: TokenStream) -> TokenStream {
    let options = match parse_metadata_options(arguments) {
        Ok(options) => options,
        Err(error) => return error.to_compile_error().into(),
    };
    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_outcome_{}_v2", function_name);
    let metadata = metadata_export(function_name, "outcome", &options)
        .unwrap_or_else(syn::Error::into_compile_error);
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
        #metadata
    }
    .into()
}

#[proc_macro_attribute]
pub fn anysearch_scenario(arguments: TokenStream, item: TokenStream) -> TokenStream {
    let options = match parse_metadata_options(arguments) {
        Ok(options) => options,
        Err(error) => return error.to_compile_error().into(),
    };
    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_scenario_{}_v1", function_name);
    let metadata = metadata_export(function_name, "scenario", &options)
        .unwrap_or_else(syn::Error::into_compile_error);
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
        #metadata
    }
    .into()
}

#[proc_macro_attribute]
pub fn anysearch_scenario_v2(arguments: TokenStream, item: TokenStream) -> TokenStream {
    let options = match parse_metadata_options(arguments) {
        Ok(options) => options,
        Err(error) => return error.to_compile_error().into(),
    };
    let function = parse_macro_input!(item as ItemFn);
    let function_name = &function.sig.ident;
    let export_name = format_ident!("anysearch_scenario_{}_v2", function_name);
    let metadata = metadata_export(function_name, "scenario", &options)
        .unwrap_or_else(syn::Error::into_compile_error);
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
        #metadata
    }.into()
}

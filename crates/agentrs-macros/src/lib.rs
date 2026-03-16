#![forbid(unsafe_code)]

//! Procedural macros for defining `agentrs` tools ergonomically.

use heck::ToUpperCamelCase;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, punctuated::Punctuated, Expr, FnArg, ItemFn, Lit, Meta, MetaNameValue, Token,
};

/// Attribute macro that turns an async function into a zero-cost tool wrapper.
#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args with Punctuated::<Meta, Token![,]>::parse_terminated);
    let function = parse_macro_input!(input as ItemFn);

    expand_tool(args.into_iter().collect(), function)
        .unwrap_or_else(|error| error.to_compile_error())
        .into()
}

fn expand_tool(args: Vec<Meta>, function: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let mut name_literal = None;
    let mut description_literal = None;

    for arg in args {
        let Meta::NameValue(MetaNameValue { path, value, .. }) = arg else {
            return Err(syn::Error::new_spanned(arg, "expected name = \"...\""));
        };

        let Expr::Lit(expr_lit) = value else {
            return Err(syn::Error::new_spanned(value, "expected string literal"));
        };
        let Lit::Str(lit) = expr_lit.lit else {
            return Err(syn::Error::new_spanned(expr_lit, "expected string literal"));
        };

        if path.is_ident("name") {
            name_literal = Some(lit);
        } else if path.is_ident("description") {
            description_literal = Some(lit);
        } else {
            return Err(syn::Error::new_spanned(path, "unsupported tool attribute"));
        }
    }

    let tool_name = name_literal
        .ok_or_else(|| syn::Error::new_spanned(&function.sig.ident, "missing tool name"))?;
    let tool_description = description_literal
        .ok_or_else(|| syn::Error::new_spanned(&function.sig.ident, "missing tool description"))?;

    if function.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            &function.sig.fn_token,
            "#[tool] requires an async function",
        ));
    }

    let function_name = &function.sig.ident;
    let visibility = &function.vis;
    let tool_struct_name = format_ident!("{}Tool", function_name.to_string().to_upper_camel_case());

    let inputs = function.sig.inputs.iter().collect::<Vec<_>>();
    if inputs.is_empty() || inputs.len() > 2 {
        return Err(syn::Error::new_spanned(
            &function.sig.inputs,
            "#[tool] expects one input argument and an optional context argument",
        ));
    }

    let input_ty = match inputs[0] {
        FnArg::Typed(argument) => &argument.ty,
        FnArg::Receiver(receiver) => {
            return Err(syn::Error::new_spanned(
                receiver,
                "tool functions cannot take self",
            ))
        }
    };

    let call_expr = if inputs.len() == 2 {
        quote! {
            let ctx = ::agentrs_tools::ToolContext::default();
            #function_name(parsed_input, &ctx).await?
        }
    } else {
        quote! {
            #function_name(parsed_input).await?
        }
    };

    Ok(quote! {
        #function

        #[derive(Debug, Clone, Default)]
        #visibility struct #tool_struct_name;

        impl #tool_struct_name {
            /// Creates the generated tool wrapper.
            #visibility fn new() -> Self {
                Self
            }
        }

        #[::agentrs_tools::__private::async_trait::async_trait]
        impl ::agentrs_core::Tool for #tool_struct_name {
            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #tool_description
            }

            fn schema(&self) -> ::agentrs_tools::__private::serde_json::Value {
                ::agentrs_tools::__private::serde_json::to_value(
                    &::agentrs_tools::__private::schemars::schema_for!(#input_ty)
                )
                .expect("tool schema should serialize")
            }

            async fn call(
                &self,
                input: ::agentrs_tools::__private::serde_json::Value,
            ) -> ::agentrs_core::Result<::agentrs_core::ToolOutput> {
                let parsed_input: #input_ty = ::agentrs_tools::__private::serde_json::from_value(input)?;
                let output = #call_expr;
                Ok(::agentrs_tools::IntoToolOutput::into_tool_output(output))
            }
        }
    })
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::CommonArgs;

#[derive(clap::Parser, Clone, Debug)]
#[command(
    name = "dynamo-trtllm-sidecar",
    about = "Run a Dynamo worker against TensorRT-LLM's native gRPC TrtllmService"
)]
pub(crate) struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    /// TensorRT-LLM gRPC endpoint as host:port or an http:// URL.
    #[arg(long, env = "TRTLLM_GRPC_ENDPOINT")]
    pub trtllm_endpoint: String,

    /// Hugging Face model ID or local path used for tokenization and templates.
    #[arg(long)]
    pub model_path: String,

    /// Model maximum sequence length (input + output). Used to register the
    /// context length and to derive a default `max_tokens` when a request omits
    /// one. TensorRT-LLM's `GetModelInfo` gRPC does not report this on current
    /// releases (it returns zero), so supply it here; otherwise requests that
    /// omit `max_tokens` are rejected. See the note in `convert.rs`.
    #[arg(long, env = "TRTLLM_CONTEXT_LENGTH")]
    pub context_length: Option<u32>,
}

// TODO(sidecar-consolidation): plaintext-scheme normalization duplicates the
// SGLang sidecar's `normalize_endpoint` (which also handles the `grpc://` alias).
// Move a parameterized `normalize_endpoint(raw, engine_label)` into the shared
// `dynamo-sidecar-grpc` crate (see the TODO in `error.rs`).
pub(crate) fn normalize_endpoint(raw: &str) -> Result<String, String> {
    let endpoint = raw.trim();
    if endpoint.is_empty() {
        return Err("TensorRT-LLM gRPC endpoint must not be empty".to_string());
    }
    if endpoint.starts_with("http://") {
        Ok(endpoint.to_string())
    } else if endpoint.contains("://") {
        Err(format!(
            "unsupported TensorRT-LLM gRPC endpoint scheme in `{endpoint}`; use plaintext http://"
        ))
    } else {
        Ok(format!("http://{endpoint}"))
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_endpoint;

    #[test]
    fn normalizes_plaintext_endpoints() {
        assert_eq!(
            normalize_endpoint(" 127.0.0.1:50051 ").unwrap(),
            "http://127.0.0.1:50051"
        );
        assert_eq!(
            normalize_endpoint("http://trtllm:50051").unwrap(),
            "http://trtllm:50051"
        );
        assert!(normalize_endpoint("https://trtllm:50051").is_err());
        assert!(normalize_endpoint(" ").is_err());
    }
}

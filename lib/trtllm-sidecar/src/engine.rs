// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for TensorRT-LLM's native `trtllm.TrtllmService` gRPC server.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest, WorkerConfig, usage,
};
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::args::{Args, normalize_endpoint};
use crate::client::TrtllmClient;
use crate::convert::{ResponseState, build_generate_request};
use crate::error;
use crate::model::ConfiguredModel;

const CONNECTIONS: usize = 8;
const ALREADY_STARTED: &str = "TensorRT-LLM sidecar has already started";

/// Terminal output emitted when a request is cancelled, carrying the usage
/// accumulated so far.
fn cancelled(state: &ResponseState) -> LLMEngineOutput {
    LLMEngineOutput::cancelled().with_usage(usage(state.prompt_tokens(), state.completion_tokens()))
}

pub struct TrtllmSidecarEngine {
    endpoint: String,
    connections: usize,
    model: ConfiguredModel,
    client: OnceCell<TrtllmClient>,
    /// Resolved model context length (`--context-length`, else `GetModelInfo`),
    /// cached at `start` so `generate` can derive a default `max_tokens` for
    /// requests that omit one.
    context_length: OnceCell<u32>,
    cancel: CancellationToken,
}

impl TrtllmSidecarEngine {
    pub(crate) fn new(endpoint: String, connections: usize, model: ConfiguredModel) -> Self {
        Self {
            endpoint,
            connections,
            model,
            client: OnceCell::new(),
            context_length: OnceCell::new(),
            cancel: CancellationToken::new(),
        }
    }

    pub fn from_env() -> Result<(Self, WorkerConfig), DynamoError> {
        Self::from_parsed(<Args as clap::Parser>::parse())
    }

    pub fn from_args(argv: Vec<String>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = <Args as clap::Parser>::try_parse_from(argv)
            .map_err(|err| error::invalid_argument(err.to_string()))?;
        Self::from_parsed(args)
    }

    fn from_parsed(args: Args) -> Result<(Self, WorkerConfig), DynamoError> {
        if args.model_path.trim().is_empty() {
            return Err(error::invalid_argument("model-path must not be empty"));
        }
        if args.common.disaggregation_mode != DisaggregationMode::Aggregated {
            return Err(error::invalid_argument(
                "the TensorRT-LLM sidecar supports aggregated serving only; the Generate \
                 response contract carries no disaggregation handoff",
            ));
        }
        if args.common.route_to_encoder {
            return Err(error::invalid_argument(
                "route-to-encoder is not supported by the TensorRT-LLM sidecar",
            ));
        }

        let endpoint =
            normalize_endpoint(&args.trtllm_endpoint).map_err(error::invalid_argument)?;
        let model = ConfiguredModel {
            source: args.model_path,
            context_length: args.context_length,
        };
        let engine = Self::new(endpoint, CONNECTIONS, model.clone());
        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            model_name: model.source.clone(),
            served_model_name: None,
            tool_call_parser: args.common.dyn_tool_call_parser,
            reasoning_parser: args.common.dyn_reasoning_parser,
            exclude_tools_when_tool_choice_none: args.common.exclude_tools_when_tool_choice_none,
            enable_kv_routing: false,
            disaggregation_mode: DisaggregationMode::Aggregated,
            route_to_encoder: false,
            ..Default::default()
        };
        Ok((engine, config))
    }
}

#[async_trait]
impl LLMEngine for TrtllmSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.client.initialized() {
            return Err(error::engine_shutdown(ALREADY_STARTED));
        }
        tracing::info!(
            endpoint = %self.endpoint,
            connections = self.connections,
            "connecting to TensorRT-LLM gRPC"
        );
        let client = TrtllmClient::connect(&self.endpoint, self.connections).await?;
        let connection_count = client.connection_count();

        // Prefer a server-reported context length; fall back to the configured
        // `--context-length`. GetModelInfo returns zero on current TRT-LLM
        // releases, so the argument is currently the only source. The resolved
        // value backs the default-`max_tokens` path in `convert::max_tokens`.
        let mut model = self.model.clone();
        match client.model_info().await {
            Ok(Some(context_length)) => model.context_length = Some(context_length),
            Ok(None) => {}
            Err(error) => tracing::warn!(%error, "GetModelInfo failed; using --context-length"),
        }
        if let Some(context_length) = model.context_length {
            let _ = self.context_length.set(context_length);
        }

        self.client
            .set(client)
            .map_err(|_| error::engine_shutdown(ALREADY_STARTED))?;
        tracing::info!(
            endpoint = %self.endpoint,
            connections = connection_count,
            model = %model.source,
            "TensorRT-LLM gRPC is ready"
        );
        Ok(model.engine_config())
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let client = self
            .client
            .get()
            .ok_or_else(|| error::engine_shutdown("TensorRT-LLM sidecar is not started"))?;
        let request_id = ctx.id().to_string();
        let proto_request =
            build_generate_request(&request, &request_id, self.context_length.get().copied())?;
        let mut state = ResponseState::new(&request);
        let cancel = self.cancel.clone();

        let stream = tokio::select! {
            biased;
            _ = ctx.stopped() => None,
            _ = cancel.cancelled() => None,
            result = client.generate(proto_request) => Some(result?),
        };
        let Some(mut stream) = stream else {
            let output = cancelled(&state);
            return Ok(Box::pin(futures::stream::once(async move { Ok(output) })));
        };

        Ok(Box::pin(async_stream::stream! {
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        yield Ok(cancelled(&state));
                        break;
                    }
                    _ = cancel.cancelled() => {
                        yield Ok(cancelled(&state));
                        break;
                    }
                    message = stream.message() => {
                        match message {
                            Ok(Some(response)) => match state.convert(response) {
                                Ok(Some(output)) => {
                                    let terminal = output.finish_reason.is_some();
                                    yield Ok(output);
                                    if terminal {
                                        break;
                                    }
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    yield Err(error);
                                    break;
                                }
                            },
                            Ok(None) => {
                                yield Err(error::protocol_error(
                                    "Generate ended before a terminal response",
                                ));
                                break;
                            }
                            Err(status) => {
                                yield Err(error::status_to_dynamo("Generate", status));
                                break;
                            }
                        }
                    }
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(client) = self.client.get() else {
            return;
        };
        if let Err(error) = client.abort(ctx.id().to_string()).await {
            tracing::debug!(request_id = ctx.id(), %error, "TensorRT-LLM Abort RPC failed");
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.cancel.cancel();
        tracing::info!("TensorRT-LLM sidecar shutdown complete");
        Ok(())
    }
}

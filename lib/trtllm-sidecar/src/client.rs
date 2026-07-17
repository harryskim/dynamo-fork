// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin client for TensorRT-LLM's native `trtllm.TrtllmService`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_backend_common::DynamoError;
use futures::future::try_join_all;
use tokio::time::Instant;
use tonic::transport::{Channel, Endpoint};

use crate::error::{
    cannot_connect, connection_timeout, invalid_argument, protocol_error, status_to_dynamo,
};
use crate::proto as pb;
use crate::proto::trtllm_service_client::TrtllmServiceClient;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const RETRY_INTERVAL: Duration = Duration::from_secs(1);
const MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
/// Deadline for the one-shot control RPCs issued at startup / on cancel, so a
/// connected-but-unresponsive server cannot hang `start` or `abort`.
const RPC_TIMEOUT: Duration = Duration::from_secs(30);

// TODO(sidecar-consolidation): the round-robin channel pool, the
// `connect_until_ready` retry loop, `MAX_MESSAGE_SIZE`, and the connect/startup
// timeout constants are engine-agnostic and duplicate the SGLang sidecar's
// `Pool`/`connect`. Hoist the generic transport into a shared `dynamo-sidecar-grpc`
// crate (see the matching TODO in `error.rs`) once a second consumer lands;
// leave only the TRT-LLM-typed `generate`/`model_info`/`abort` methods here.
/// Fixed-size pool of independent HTTP/2 connections. Generation calls are
/// round-robined so high concurrency does not funnel through one codec task.
pub(crate) struct TrtllmClient {
    channels: Vec<Channel>,
    next: AtomicUsize,
}

impl TrtllmClient {
    pub(crate) async fn connect(endpoint: &str, connections: usize) -> Result<Self, DynamoError> {
        if connections == 0 {
            return Err(invalid_argument(
                "TensorRT-LLM gRPC connection count must be greater than zero",
            ));
        }
        let endpoint = Endpoint::from_shared(endpoint.to_string())
            .map_err(|error| invalid_argument(format!("invalid TensorRT-LLM endpoint: {error}")))?
            .connect_timeout(CONNECT_TIMEOUT);
        let first = connect_until_ready(endpoint.clone()).await?;
        let mut channels = vec![first];
        let remaining = try_join_all((1..connections).map(|_| {
            let endpoint = endpoint.clone();
            async move { connect_until_ready(endpoint).await }
        }))
        .await?;
        channels.extend(remaining);
        Ok(Self {
            channels,
            next: AtomicUsize::new(0),
        })
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.channels.len()
    }

    fn client(&self) -> TrtllmServiceClient<Channel> {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.channels.len();
        TrtllmServiceClient::new(self.channels[index].clone())
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
    }

    pub(crate) async fn generate(
        &self,
        request: pb::GenerateRequest,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, DynamoError> {
        self.client()
            .generate(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| status_to_dynamo("Generate", status))
    }

    /// Queries `GetModelInfo` and returns the reported maximum sequence length
    /// (input + output) as the registration context length, if positive.
    pub(crate) async fn model_info(&self) -> Result<Option<u32>, DynamoError> {
        let info = tokio::time::timeout(
            RPC_TIMEOUT,
            self.client().get_model_info(pb::GetModelInfoRequest {}),
        )
        .await
        .map_err(|_| {
            connection_timeout(format!(
                "GetModelInfo did not respond within {RPC_TIMEOUT:?}"
            ))
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetModelInfo", status))?;
        Ok(u32::try_from(info.max_seq_len).ok().filter(|len| *len > 0))
    }

    pub(crate) async fn abort(&self, request_id: String) -> Result<(), DynamoError> {
        let response = tokio::time::timeout(
            RPC_TIMEOUT,
            self.client().abort(pb::AbortRequest { request_id }),
        )
        .await
        .map_err(|_| connection_timeout(format!("Abort did not respond within {RPC_TIMEOUT:?}")))?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("Abort", status))?;
        if !response.success {
            return Err(protocol_error(format!(
                "TensorRT-LLM rejected abort: {}",
                response.message
            )));
        }
        Ok(())
    }
}

async fn connect_until_ready(endpoint: Endpoint) -> Result<Channel, DynamoError> {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        match endpoint.clone().connect().await {
            Ok(channel) => return Ok(channel),
            Err(error) if Instant::now() < deadline => {
                tracing::debug!(%error, "waiting for TensorRT-LLM gRPC");
                tokio::time::sleep(RETRY_INTERVAL).await;
            }
            Err(error) => {
                return Err(cannot_connect(format!(
                    "failed to connect to TensorRT-LLM within {STARTUP_TIMEOUT:?}: {error}"
                )));
            }
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `DynamoError` constructors and gRPC status mapping for the sidecar.
//!
//! These helpers are engine-agnostic: they wrap `dynamo_backend_common`'s
//! `BackendError` taxonomy and translate `tonic::Status` codes.
//!
//! TODO(sidecar-consolidation): this module is the cleanest lift-and-shift seam.
//! The SGLang sidecar carries a near-identical set (already drifting — its
//! `status_to_dynamo` does not map `FailedPrecondition`/`AlreadyExists`). When a
//! second consumer lands (the `vllm-sidecar` branches make three), move this
//! module verbatim into a shared `dynamo-sidecar-grpc` crate, together with the
//! transport pool (see the TODO in `client.rs`) and `normalize_endpoint` (see the
//! TODO in `args.rs`). The tonic-free constructors could alternatively live on
//! `dynamo_backend_common::error`, but `status_to_dynamo` pulls in tonic, which
//! backend-common intentionally does not depend on today.

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};

fn backend(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> DynamoError {
    backend(BackendError::InvalidArgument, message)
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    backend(
        BackendError::Unknown,
        format!("invalid TensorRT-LLM gRPC response: {}", message.into()),
    )
}

pub(crate) fn engine_shutdown(message: impl Into<String>) -> DynamoError {
    backend(BackendError::EngineShutdown, message)
}

pub(crate) fn cannot_connect(message: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, message)
}

pub(crate) fn status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    let kind = match status.code() {
        tonic::Code::InvalidArgument
        | tonic::Code::NotFound
        | tonic::Code::OutOfRange
        | tonic::Code::FailedPrecondition
        | tonic::Code::AlreadyExists => BackendError::InvalidArgument,
        tonic::Code::Unavailable => BackendError::CannotConnect,
        tonic::Code::Cancelled => BackendError::Cancelled,
        tonic::Code::DeadlineExceeded => BackendError::ConnectionTimeout,
        _ => BackendError::Unknown,
    };
    backend(
        kind,
        format!("{rpc}: {} ({:?})", status.message(), status.code()),
    )
}

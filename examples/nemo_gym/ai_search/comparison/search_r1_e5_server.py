# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Search-R1 E5 retriever with bounded loading and metrics.

The official 21-million-vector index briefly exceeds an 80 GB H100 when the
upstream helper clones it in one call. This launcher keeps the same float16
FlatIP search and upstream corpus/model code, but transfers the index in bounded
chunks. It also exposes low-cardinality Prometheus metrics at ``/metrics``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from resources_servers.ai_search.observability import trace_span
from retrieval_serialization import SerializedRequestGate


_GPU_RESOURCES: list[object] = []
_CLONE_SECONDS = 0.0
_LATENCY_BUCKETS_SECONDS = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    30,
)


class QueryRequest(BaseModel):
    """Search-R1-compatible retrieval request."""

    queries: list[str] = Field(min_length=1)
    topk: int | None = Field(default=None, ge=1)
    return_scores: bool = False


class E5PrometheusMetrics:
    """Prometheus registry for one official E5 service process."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "search_r1_e5_requests_total",
            "Official Search-R1 E5 requests by outcome.",
            ("outcome",),
            registry=self.registry,
        )
        self.queries = Counter(
            "search_r1_e5_queries_total",
            "Queries processed by the official Search-R1 E5 service.",
            registry=self.registry,
        )
        self.inflight = Gauge(
            "search_r1_e5_requests_inflight",
            "E5 retrieval requests currently executing.",
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "search_r1_e5_batch_size",
            "Queries in each E5 encoder/index batch.",
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
            registry=self.registry,
        )
        self.stage_seconds = Histogram(
            "search_r1_e5_stage_seconds",
            "Wall-clock E5 retrieval time by stage.",
            ("stage",),
            buckets=_LATENCY_BUCKETS_SECONDS,
            registry=self.registry,
        )
        self.index_vectors = Gauge(
            "search_r1_e5_index_vectors",
            "Vectors loaded into the official Search-R1 E5 GPU index.",
            registry=self.registry,
        )
        self.clone_seconds = Gauge(
            "search_r1_e5_index_clone_seconds",
            "Seconds required to copy the official index to the GPU.",
            registry=self.registry,
        )
        for outcome in ("success", "error"):
            self.requests.labels(outcome=outcome)
        for stage in ("queue", "encode", "index", "fetch", "request_total"):
            self.stage_seconds.labels(stage=stage)


def _chunked_index_cpu_to_all_gpus(index, co=None, ngpu: int = -1):
    """Clone the official FlatIP index to one GPU in bounded-size chunks."""
    global _CLONE_SECONDS

    gpu_count = faiss.get_num_gpus()
    if gpu_count != 1:
        raise RuntimeError(
            "The bounded Search-R1 E5 loader requires exactly one visible GPU, "
            f"but Faiss sees {gpu_count}."
        )
    if ngpu not in (-1, 1):
        raise RuntimeError(f"Unsupported ngpu={ngpu}; expected -1 or 1.")
    if co is None or not bool(co.useFloat16) or not bool(co.shard):
        raise RuntimeError(
            "The upstream Search-R1 loader must request float16 sharded cloning."
        )
    if type(index).__name__ != "IndexFlatIP":
        raise RuntimeError(
            "The bounded loader only supports the official IndexFlatIP, not "
            f"{type(index).__name__}."
        )
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise RuntimeError("The official Search-R1 E5 index must use inner product.")

    expected_vectors = int(os.environ.get("SEARCH_R1_E5_EXPECTED_VECTORS", "21015324"))
    expected_dimension = int(os.environ.get("SEARCH_R1_E5_EXPECTED_DIMENSION", "768"))
    if index.ntotal != expected_vectors or index.d != expected_dimension:
        raise RuntimeError(
            "Unexpected Search-R1 E5 index shape: "
            f"ntotal={index.ntotal}, d={index.d}; expected "
            f"ntotal={expected_vectors}, d={expected_dimension}."
        )

    chunk_size = int(os.environ.get("SEARCH_R1_E5_CLONE_CHUNK_VECTORS", "131072"))
    temp_memory_mib = int(os.environ.get("SEARCH_R1_E5_FAISS_TEMP_MEMORY_MIB", "512"))
    if chunk_size <= 0 or temp_memory_mib <= 0:
        raise RuntimeError("Chunk size and Faiss temporary memory must be positive.")

    resources = faiss.StandardGpuResources()
    resources.setTempMemory(temp_memory_mib * 1024 * 1024)
    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    config.useFloat16 = True
    gpu_index = faiss.GpuIndexFlatIP(resources, index.d, config)

    started = time.perf_counter()
    for start in range(0, index.ntotal, chunk_size):
        count = min(chunk_size, index.ntotal - start)
        vectors = np.ascontiguousarray(index.reconstruct_n(start, count))
        gpu_index.add(vectors)
        if (
            start == 0
            or gpu_index.ntotal == index.ntotal
            or (start // chunk_size) % 16 == 0
        ):
            print(
                f"search_r1_e5_clone_progress={gpu_index.ntotal}/{index.ntotal}",
                flush=True,
            )

    if gpu_index.ntotal != index.ntotal:
        raise RuntimeError(
            f"Incomplete GPU clone: {gpu_index.ntotal}/{index.ntotal} vectors."
        )

    _GPU_RESOURCES.append(resources)
    _CLONE_SECONDS = time.perf_counter() - started
    print(
        "SEARCH_R1_E5_CHUNKED_CLONE_READY "
        f"vectors={gpu_index.ntotal} dimension={gpu_index.d} "
        f"seconds={_CLONE_SECONDS:.3f}",
        flush=True,
    )
    return gpu_index


def _load_upstream(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Missing upstream retrieval server: {path}")
    spec = importlib.util.spec_from_file_location("search_r1_retrieval_server", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream retrieval server: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _timed_batch_search(
    *,
    retriever: Any,
    upstream: ModuleType,
    queries: list[str],
    top_k: int,
    metrics: E5PrometheusMetrics,
) -> tuple[list[list[dict[str, Any]]], list[list[float]], dict[str, float]]:
    """Execute the upstream E5 algorithm while timing its three real stages."""
    results: list[list[dict[str, Any]]] = []
    scores: list[list[float]] = []
    totals = {"encode": 0.0, "index": 0.0, "fetch": 0.0}

    for start in range(0, len(queries), retriever.batch_size):
        query_batch = queries[start : start + retriever.batch_size]
        metrics.batch_size.observe(len(query_batch))

        stage_started = time.perf_counter()
        batch_embeddings = retriever.encoder.encode(query_batch)
        elapsed = time.perf_counter() - stage_started
        totals["encode"] += elapsed
        metrics.stage_seconds.labels(stage="encode").observe(elapsed)

        stage_started = time.perf_counter()
        batch_scores, batch_indices = retriever.index.search(batch_embeddings, k=top_k)
        elapsed = time.perf_counter() - stage_started
        totals["index"] += elapsed
        metrics.stage_seconds.labels(stage="index").observe(elapsed)

        stage_started = time.perf_counter()
        score_rows = batch_scores.tolist()
        index_rows = batch_indices.tolist()
        flat_indices = sum(index_rows, [])
        batch_results = upstream.load_docs(retriever.corpus, flat_indices)
        batch_results = [
            batch_results[index * top_k : (index + 1) * top_k]
            for index in range(len(index_rows))
        ]
        elapsed = time.perf_counter() - stage_started
        totals["fetch"] += elapsed
        metrics.stage_seconds.labels(stage="fetch").observe(elapsed)

        results.extend(batch_results)
        scores.extend(score_rows)
        del (
            batch_embeddings,
            batch_scores,
            batch_indices,
            query_batch,
            flat_indices,
            batch_results,
        )
        torch.cuda.empty_cache()

    return results, scores, totals


def _create_app(
    *, retriever: Any, upstream: ModuleType, metrics: E5PrometheusMetrics
) -> FastAPI:
    app = FastAPI(title="Observed official Search-R1 E5 retriever")
    request_gate = SerializedRequestGate()

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ready",
            "vectors": int(retriever.index.ntotal),
            "dimension": int(retriever.index.d),
            "max_concurrent_requests": 1,
        }

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post("/retrieve")
    def retrieve_endpoint(body: QueryRequest, request: Request) -> dict[str, Any]:
        top_k = body.topk or retriever.topk
        batch_id = request.headers.get("X-NeMo-Search-Batch-ID") or uuid.uuid4().hex
        metrics.inflight.inc()
        request_started = time.perf_counter()
        with trace_span(
            trace_id=f"retriever-batch:{batch_id}",
            component="search_r1_e5",
            operation="retrieve",
            attributes={
                "provider_batch_id": batch_id,
                "queries": len(body.queries),
                "top_k": top_k,
            },
        ) as span:
            try:
                with request_gate.enter() as queue_seconds:
                    metrics.stage_seconds.labels(stage="queue").observe(
                        queue_seconds
                    )
                    results, scores, stage_totals = _timed_batch_search(
                        retriever=retriever,
                        upstream=upstream,
                        queries=body.queries,
                        top_k=top_k,
                        metrics=metrics,
                    )
                response_rows: list[list[Any]] = []
                for result_row, score_row in zip(results, scores):
                    if body.return_scores:
                        response_rows.append(
                            [
                                {"document": document, "score": score}
                                for document, score in zip(result_row, score_row)
                            ]
                        )
                    else:
                        response_rows.append(result_row)
                request_seconds = time.perf_counter() - request_started
                metrics.requests.labels(outcome="success").inc()
                metrics.queries.inc(len(body.queries))
                metrics.stage_seconds.labels(stage="request_total").observe(
                    request_seconds
                )
                span.set_attributes(
                    outcome="success",
                    request_ms=request_seconds * 1000.0,
                    queue_ms=queue_seconds * 1000.0,
                    encode_ms=stage_totals["encode"] * 1000.0,
                    index_ms=stage_totals["index"] * 1000.0,
                    fetch_ms=stage_totals["fetch"] * 1000.0,
                )
                return {"result": response_rows}
            except Exception:
                metrics.requests.labels(outcome="error").inc()
                raise
            finally:
                metrics.inflight.dec()

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-server", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--retriever-model", type=Path, required=True)
    parser.add_argument("--retriever-name", default="e5")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--faiss-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for path in (args.index_path, args.corpus_path, args.retriever_model):
        if not path.exists():
            raise FileNotFoundError(f"Missing required E5 artifact: {path}")
    if args.topk <= 0:
        raise ValueError(f"--topk must be positive, not {args.topk}")

    faiss.index_cpu_to_all_gpus = _chunked_index_cpu_to_all_gpus
    upstream = _load_upstream(args.upstream_server)
    config = upstream.Config(
        retrieval_method=args.retriever_name,
        index_path=str(args.index_path),
        corpus_path=str(args.corpus_path),
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=str(args.retriever_model),
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    retriever = upstream.get_retriever(config)
    metrics = E5PrometheusMetrics()
    metrics.index_vectors.set(int(retriever.index.ntotal))
    metrics.clone_seconds.set(_CLONE_SECONDS)
    app = _create_app(retriever=retriever, upstream=upstream, metrics=metrics)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

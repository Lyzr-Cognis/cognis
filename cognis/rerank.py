"""Small fail-open client for compatible remote reranking endpoints."""

import json
import logging
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class RemoteReranker:
    """Rerank documents remotely without making remote search a dependency."""

    def __init__(self, url: str, api_key: Optional[str] = None, timeout_seconds: float = 5.0):
        # "sagemaker://<endpoint-name>" routes through sagemaker-runtime
        # invoke_endpoint (TEI container); anything else is a plain HTTP
        # TEI-compatible /rerank endpoint.
        self.sagemaker_endpoint: Optional[str] = None
        self._sagemaker_client = None
        if url.startswith("sagemaker://"):
            self.sagemaker_endpoint = url[len("sagemaker://"):].strip("/")
            self.url = url
        else:
            base_url = url.rstrip("/")
            self.url = base_url if base_url.endswith("/rerank") else f"{base_url}/rerank"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _invoke_sagemaker(self, body: bytes) -> Any:
        if self._sagemaker_client is None:
            import boto3
            from botocore.config import Config

            self._sagemaker_client = boto3.client(
                "sagemaker-runtime",
                config=Config(
                    read_timeout=self.timeout_seconds,
                    connect_timeout=min(10.0, self.timeout_seconds),
                    retries={"max_attempts": 1},
                ),
            )
        response = self._sagemaker_client.invoke_endpoint(
            EndpointName=self.sagemaker_endpoint,
            ContentType="application/json",
            Body=body,
        )
        return json.loads(response["Body"].read().decode("utf-8"))

    def rerank(self, query: str, documents: List[str]) -> Optional[List[float]]:
        """Return scores aligned with documents, or None when the endpoint fails."""
        if not documents:
            return []
        # SageMaker TEI containers cap client batch size (default 32); chunk and
        # stitch. HTTP backends take the full list in one request.
        chunk = 32 if self.sagemaker_endpoint else len(documents)
        if len(documents) > chunk:
            scores: List[float] = []
            for start in range(0, len(documents), chunk):
                part = self._rerank_once(query, documents[start : start + chunk])
                if part is None:
                    return None
                scores.extend(part)
            return scores
        return self._rerank_once(query, documents)

    def _rerank_once(self, query: str, documents: List[str]) -> Optional[List[float]]:
        # Dual schema: local reranker_server reads "documents"; HF TEI reads
        # "texts" (and ignores unknown fields). Send both so either backend works.
        payload = json.dumps(
            {"query": query, "documents": documents, "texts": documents}
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            if self.sagemaker_endpoint:
                data: Any = self._invoke_sagemaker(payload)
            else:
                request = Request(self.url, data=payload, headers=headers, method="POST")
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
            # TEI returns a bare list [{"index": i, "score": s}]; the local
            # server wraps it as {"results": [...{"relevance_score": s}]}.
            if isinstance(data, list):
                results = data
            else:
                results = data.get("results", data.get("data", data.get("scores")))
            if not isinstance(results, list):
                raise ValueError("rerank response has no result list")
            scores: List[Optional[float]] = [None] * len(documents)
            for position, item in enumerate(results):
                if isinstance(item, (int, float)) and position < len(scores):
                    scores[position] = float(item)
                elif isinstance(item, dict):
                    index = item.get("index", position)
                    score = item.get("relevance_score", item.get("score"))
                    if isinstance(index, int) and 0 <= index < len(scores) and isinstance(score, (int, float)):
                        scores[index] = float(score)
            if any(score is None for score in scores):
                raise ValueError("rerank response did not score every document")
            return [float(score) for score in scores]
        except Exception as exc:
            logger.warning("Remote reranking failed; returning RRF order: %s", exc)
            return None

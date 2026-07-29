from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import AppConfig
from .generated_features import (
    DEFAULT_GENERATED_FEATURE_INPUTS,
    FEATURE_FILE_SCHEMA_VERSION,
    FeatureDefinition,
    parse_feature_candidates,
)


@dataclass(frozen=True)
class DeepSeekGenerationResult:
    features: list[FeatureDefinition]
    rejected: list[dict[str, Any]]
    model: str
    response_id: str | None
    usage: dict[str, Any]


def build_feature_generation_prompt(proposal_count: int) -> tuple[str, str]:
    allowed_fields = ", ".join(sorted(DEFAULT_GENERATED_FEATURE_INPUTS))
    system_prompt = """
You are a quantitative equity feature researcher. Generate economically motivated,
point-in-time-safe features for daily cross-sectional stock selection. Return only
one valid JSON object. Never use future data, targets, forward returns, negative
lags, imports, attribute access, indexing, Python statements, or arbitrary code.
The local system will parse a restricted expression DSL and automatically apply a
within-date cross-sectional percentile rank to every proposed feature.
""".strip()
    user_prompt = f"""
Generate exactly {proposal_count} diverse candidate features for an A-share
five-trading-day cross-sectional return-ranking model.

Allowed input fields:
{allowed_fields}

Allowed expression syntax:
- arithmetic: +, -, *, /
- unary signs: +x, -x
- scalar constants
- abs(x)
- signed_log1p(x)
- safe_div(x, y)
- minimum(x, y), maximum(x, y)
- clip(x, lower_constant, upper_constant)
- lag(x, n), delta(x, n), pct_change(x, n)
- rolling_mean/std/min/max/sum(x, n)
- ema(x, n)
- ts_zscore(x, n)

All window values must be positive integer constants no larger than 252.
Generated names must start with ai_ and match ai_[a-z][a-z0-9_]*.
Do not call cs_rank: the pipeline applies it automatically.
Do not merely duplicate a single allowed input. Prefer economically meaningful
interactions, volatility normalization, liquidity conditioning, regime comparison,
multi-horizon trend/reversal, overnight/intraday decomposition, and size-aware
microstructure signals. Avoid producing near-duplicate formulas.

Return JSON in exactly this shape:
{{
  "features": [
    {{
      "name": "ai_example_feature",
      "expression": "safe_div(momentum_5 - momentum_20, volatility_20)",
      "rationale": "At least one concise sentence explaining the economic idea.",
      "expected_direction": "positive",
      "source": "deepseek"
    }}
  ]
}}

expected_direction must be positive, negative, or unknown. The final response must
be valid JSON with no Markdown fences and no commentary outside the JSON object.
""".strip()
    return system_prompt, user_prompt


class DeepSeekFeatureGenerator:
    def __init__(
        self,
        config: AppConfig,
        *,
        opener: Callable[..., Any] = urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config.deepseek
        self.opener = opener
        self.sleep = sleep

    def _api_key(self) -> str:
        token_name = self.config.api_key_env
        token = os.getenv(token_name, "").strip()
        if not token:
            raise RuntimeError(
                f"DeepSeek API key is missing. Set it with "
                f"export {token_name}='your-key'"
            )
        return token

    def _request_payload(self, proposal_count: int) -> dict[str, Any]:
        system_prompt, user_prompt = build_feature_generation_prompt(proposal_count)
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {
                "type": (
                    "enabled" if self.config.thinking_enabled else "disabled"
                )
            },
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            endpoint,
            data=encoded,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ashare-stock-selection/0.2",
            },
        )
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                with self.opener(
                    request,
                    timeout=self.config.timeout_seconds,
                ) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise RuntimeError("DeepSeek returned a non-object response")
                choices = parsed.get("choices")
                message = (
                    choices[0].get("message")
                    if isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                    else None
                )
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("DeepSeek returned empty response content")
                return parsed
            except HTTPError as error:
                last_error = error
                if error.code < 500 and error.code != 429:
                    body = error.read().decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(
                        f"DeepSeek API returned HTTP {error.code}: {body}"
                    ) from error
            except (URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as error:
                last_error = error
            if attempt < self.config.max_retries:
                self.sleep(self.config.retry_backoff_seconds * attempt)
        raise RuntimeError(
            f"DeepSeek API failed after {self.config.max_retries} attempts: "
            f"{last_error}"
        ) from last_error

    def generate(self, proposal_count: int | None = None) -> DeepSeekGenerationResult:
        count = (
            self.config.proposal_count
            if proposal_count is None
            else int(proposal_count)
        )
        if count < 1 or count > 100:
            raise ValueError("proposal_count must be in [1, 100]")
        response = self._post(self._request_payload(count))
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("DeepSeek response contains no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("DeepSeek choice has an invalid format")
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise RuntimeError(
                "DeepSeek JSON was truncated; increase deepseek.max_tokens"
            )
        if finish_reason not in {None, "stop"}:
            raise RuntimeError(
                f"DeepSeek generation did not complete normally: {finish_reason}"
            )
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek returned empty feature JSON")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"DeepSeek returned invalid feature JSON: {error}") from error
        features, rejected = parse_feature_candidates(
            payload,
            allowed_names=DEFAULT_GENERATED_FEATURE_INPUTS,
        )
        if not features:
            reasons = "; ".join(item["reason"] for item in rejected[:5])
            raise RuntimeError(
                f"DeepSeek produced no valid feature definitions: {reasons}"
            )
        usage = response.get("usage")
        return DeepSeekGenerationResult(
            features=features,
            rejected=rejected,
            model=str(response.get("model") or self.config.model),
            response_id=(
                str(response["id"]) if response.get("id") is not None else None
            ),
            usage=dict(usage) if isinstance(usage, dict) else {},
        )


def write_deepseek_generation_result(
    result: DeepSeekGenerationResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FEATURE_FILE_SCHEMA_VERSION,
        "generator": {
            "provider": "deepseek",
            "model": result.model,
            "response_id": result.response_id,
            "usage": result.usage,
        },
        "features": [feature.to_dict() for feature in result.features],
        "rejected": result.rejected,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path

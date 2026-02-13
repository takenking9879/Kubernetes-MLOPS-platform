import random
from typing import Any, Dict


class TrafficRouter:
    """Pure canary traffic router for inference payloads."""

    def __init__(self, stable_handle, canary_handle, canary_probability: float):
        self._stable = stable_handle
        self._canary = canary_handle
        self._canary_probability = self._clamp_probability(canary_probability)

    @staticmethod
    def _clamp_probability(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def set_canary_probability(self, value: float) -> None:
        self._canary_probability = self._clamp_probability(value)

    @property
    def canary_probability(self) -> float:
        return self._canary_probability

    async def route(self, payload: Dict[str, Any]):
        use_canary = random.random() < self._canary_probability
        if use_canary and self._canary is not None:
            return await self._canary.predict.remote(payload)
        return await self._stable.predict.remote(payload)

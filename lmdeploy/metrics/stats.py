# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/stats.py

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from lmdeploy.messages import EngineCoreEvent, ResponseType
    from lmdeploy.pytorch.engine import InferOutput


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    gpu_cache_usage: float = 0.0

    def __repr__(self):
        """Return a human-readable string representation."""
        return (f'SchedulerStats(\n'
                f'  num_running_reqs={self.num_running_reqs},\n'
                f'  num_waiting_reqs={self.num_waiting_reqs},\n'
                f'  gpu_cache_usage={self.gpu_cache_usage:.6f},\n'
                f')')


@dataclass
class RequestStateStats:
    """Stats that need to be tracked across delta updates."""

    num_generation_tokens: int = 0

    # This is a engine frontend timestamp
    arrival_time: float = 0.0

    # These are engine core timestamps
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0


@dataclass
class FinishedRequestStats:
    """Stats associated with a finished request."""

    finish_reason: 'ResponseType'
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    queued_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0


class IterationStats:
    """Stats associated with a single set of EngineCoreOutputs."""

    def __init__(self):
        self.iteration_timestamp = time.perf_counter()
        self.num_generation_tokens = 0
        self.num_prompt_tokens = 0
        self.finished_requests: list[FinishedRequestStats] = []
        self.time_to_first_tokens_iter: list[float] = []
        self.time_per_output_tokens_iter: list[float] = []

    def __repr__(self):
        """Return a human-readable string representation."""
        return (f'IterationStats(\n'
                f'  iteration_timestamp={self.iteration_timestamp:.6f},\n'
                f'  num_generation_tokens={self.num_generation_tokens},\n'
                f'  num_prompt_tokens={self.num_prompt_tokens},\n'
                f'  finished_requests_count={len(self.finished_requests)},\n'
                f'  time_to_first_tokens_iter={self.time_to_first_tokens_iter},\n'
                f'  time_per_output_tokens_iter={self.time_per_output_tokens_iter},\n'
                f')')

    def _time_since(self, start: float) -> float:
        """Calculate an interval relative to this iteration's timestamp."""
        return self.iteration_timestamp - start

    def update_from_output(self, output: 'InferOutput', engine_core_timestamp: float, is_prefilling: bool,
                           prompt_len: int, req_stats: RequestStateStats):
        num_new_generation_tokens = 1

        self.num_generation_tokens += num_new_generation_tokens
        if is_prefilling:
            assert num_new_generation_tokens > 0
            self.num_prompt_tokens += prompt_len

            first_token_latency = self._time_since(req_stats.arrival_time)
            self.time_to_first_tokens_iter.append(first_token_latency)

        req_stats.num_generation_tokens += num_new_generation_tokens

        # Process request-level engine core events
        if output.engine_core_events is not None:
            self.update_from_events(output.engine_core_events, req_stats)

        # Process the batch-level "new tokens" engine core event
        if is_prefilling:
            req_stats.first_token_ts = engine_core_timestamp
        else:
            tpot = engine_core_timestamp - req_stats.last_token_ts
            self.time_per_output_tokens_iter.append(tpot)

        req_stats.last_token_ts = engine_core_timestamp

    def update_from_events(self, engine_core_events: List['EngineCoreEvent'], req_stats: RequestStateStats):
        # Avoid circular dependency
        from lmdeploy.messages import EngineCoreEventType

        for event in engine_core_events:
            if event.type == EngineCoreEventType.QUEUED:
                req_stats.queued_ts = event.timestamp
            elif event.type == EngineCoreEventType.SCHEDULED:
                if req_stats.scheduled_ts == 0.0:  # ignore preemptions
                    req_stats.scheduled_ts = event.timestamp
            # FIXME: deal with preempted case
            # elif event.type == EngineCoreEventType.PREEMPTED:
            #     self.num_preempted_reqs += 1

    def update_from_finished_request(self, finish_reason: 'ResponseType', num_prompt_tokens: int,
                                     req_stats: RequestStateStats):

        e2e_latency = self._time_since(req_stats.arrival_time)

        # Queued interval is from first QUEUED event to first SCHEDULED
        queued_time = req_stats.scheduled_ts - req_stats.queued_ts

        # Prefill interval is from first SCHEDULED to first NEW_TOKEN
        # Any preemptions during prefill is included in the interval
        prefill_time = req_stats.first_token_ts - req_stats.scheduled_ts

        # Decode interval is from first NEW_TOKEN to last NEW_TOKEN
        # Any preemptions during decode are included
        decode_time = req_stats.last_token_ts - req_stats.first_token_ts

        # Inference interval is from first SCHEDULED to last NEW_TOKEN
        # Any preemptions during prefill or decode are included
        inference_time = req_stats.last_token_ts - req_stats.scheduled_ts

        finished_req = \
            FinishedRequestStats(finish_reason=finish_reason,
                                 e2e_latency=e2e_latency,
                                 num_prompt_tokens=num_prompt_tokens,
                                 num_generation_tokens=req_stats.num_generation_tokens,
                                 queued_time=queued_time,
                                 prefill_time=prefill_time,
                                 inference_time=inference_time,
                                 decode_time=decode_time)
        self.finished_requests.append(finished_req)

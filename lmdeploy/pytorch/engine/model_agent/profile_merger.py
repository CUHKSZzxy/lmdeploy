# Copyright (c) OpenMMLab. All rights reserved.
import gzip
import json
import os
from typing import Any, TextIO


class ProfileMerger:
    """Merge per-rank PyTorch profiler traces into one Chrome trace."""

    _RANK_SORT_INDEX_STRIDE = 1_000_000_000

    def __init__(self, output_prefix: str, world_size: int, use_gzip: bool):
        self.output_prefix = output_prefix
        self.world_size = world_size
        self.use_gzip = use_gzip
        self.suffix = '.json.gz' if use_gzip else '.json'
        self.output_path = f'{output_prefix}merged{self.suffix}'

    def merge(self) -> tuple[str, int]:
        """Merge all expected rank traces and return path and event count."""
        temp_path = f'{self.output_path}.tmp'
        self._remove_file(self.output_path)
        try:
            event_count = self._merge_to_path(temp_path)
            os.replace(temp_path, self.output_path)
        except Exception:
            self._remove_file(temp_path)
            raise
        return self.output_path, event_count

    def _merge_to_path(self, output_path: str) -> int:
        first_trace = self._load_trace(self._rank_path(0))
        metadata = {
            key: value
            for key, value in first_trace.items()
            if key not in ('traceEvents', 'deviceProperties', 'traceName')
        }
        metadata['traceName'] = self.output_path
        base_time = first_trace.get('baseTimeNanoseconds')
        device_properties = []
        event_count = 0

        with self._open_text(output_path, 'wt') as output:
            output.write('{')
            for index, (key, value) in enumerate(metadata.items()):
                if index:
                    output.write(',')
                json.dump(key, output)
                output.write(':')
                json.dump(value, output, separators=(',', ':'))

            if metadata:
                output.write(',')
            output.write('"traceEvents":[')

            first_event = True
            for rank in range(self.world_size):
                trace = first_trace if rank == 0 else self._load_trace(self._rank_path(rank))
                self._validate_metadata(trace, metadata, rank)
                timestamp_offset = self._get_timestamp_offset(trace, base_time, rank)
                rank_device_properties = trace.get('deviceProperties', [])
                if not isinstance(rank_device_properties, list):
                    raise ValueError(f'Profile trace rank {rank} has invalid deviceProperties.')
                device_properties.extend(rank_device_properties)

                for event in trace['traceEvents']:
                    if not first_event:
                        output.write(',')
                    json.dump(self._process_event(event, rank, timestamp_offset),
                              output,
                              separators=(',', ':'))
                    first_event = False
                    event_count += 1

            output.write(']')
            if device_properties:
                output.write(',"deviceProperties":')
                json.dump(device_properties, output, separators=(',', ':'))
            output.write('}')

        return event_count

    def _load_trace(self, path: str) -> dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing profile trace: {path}')
        with self._open_text(path, 'rt') as trace_file:
            trace = json.load(trace_file)
        if not isinstance(trace, dict) or not isinstance(trace.get('traceEvents'), list):
            raise ValueError(f'Profile trace has invalid traceEvents: {path}')
        return trace

    @staticmethod
    def _validate_metadata(trace: dict[str, Any], metadata: dict[str, Any], rank: int):
        for key in ('schemaVersion', 'displayTimeUnit'):
            if trace.get(key) != metadata.get(key):
                raise ValueError(f'Profile trace rank {rank} has inconsistent {key}.')

    @staticmethod
    def _get_timestamp_offset(trace: dict[str, Any], base_time: Any, rank: int) -> float:
        rank_base_time = trace.get('baseTimeNanoseconds')
        if rank_base_time == base_time:
            return 0.0
        if rank_base_time is None or base_time is None:
            raise ValueError(f'Profile trace rank {rank} has inconsistent baseTimeNanoseconds.')
        return (rank_base_time - base_time) / 1000

    def _process_event(self, event: Any, rank: int, timestamp_offset: float) -> dict[str, Any]:
        if not isinstance(event, dict):
            raise ValueError(f'Profile trace rank {rank} contains a non-object event.')

        event = event.copy()
        if timestamp_offset and 'ts' in event:
            event['ts'] += timestamp_offset
        if 'pid' in event:
            event['pid'] = f'[rank{rank}] {event["pid"]}'

        name = event.get('name')
        if name in ('process_name', 'process_sort_index'):
            args = event.get('args', {}).copy()
            if name == 'process_name' and 'name' in args:
                args['name'] = f'[rank{rank}] {args["name"]}'
            elif name == 'process_sort_index' and 'sort_index' in args:
                args['sort_index'] += rank * self._RANK_SORT_INDEX_STRIDE
            event['args'] = args
        return event

    def _rank_path(self, rank: int) -> str:
        return f'{self.output_prefix}{rank}{self.suffix}'

    def _open_text(self, path: str, mode: str) -> TextIO:
        if self.use_gzip:
            return gzip.open(path, mode, encoding='utf-8')
        return open(path, mode, encoding='utf-8')

    @staticmethod
    def _remove_file(path: str):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

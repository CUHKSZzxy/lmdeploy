# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os

import torch
from torch import distributed as dist
from torch.profiler import ProfilerActivity, profile

from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.utils import get_logger

from .profile_merger import ProfileMerger

logger = get_logger('lmdeploy')


class AgentProfiler:

    def __init__(self, dist_ctx: DistContext, stream: torch.Stream):
        from lmdeploy.pytorch import envs
        self.rank = dist_ctx.rank
        self.dp_rank = dist_ctx.dp_rank
        self.dp = dist_ctx.dist_config.dp
        self.world_size = dist_ctx.dist_config.world_size
        self.cpu_group = dist_ctx.cpu_group
        self.stream = stream
        self.profiler = None
        self.name = f'rank[{self.rank}]'

        self.delay = envs.torch_profile_delay
        self.duration = envs.torch_profile_duration

        self.profiler = self._build_profiler()
        self.prefix = envs.torch_profile_output_prefix
        self.use_gzip = envs.torch_profile_use_gzip
        self.merge_profiles = envs.torch_profile_merge
        self._task = None
        self._started = False
        if self.merge_profiles and self.duration <= 0:
            logger.warning('LMDEPLOY_PROFILE_MERGE requires LMDEPLOY_PROFILE_DURATION > 0. '
                           'Profile merging is disabled for this capture.')
            self.merge_profiles = False
        if self.dp > 1 and self.duration < 0 and self.profiler is not None:
            logger.warning('Do not support duration<=0 for dp > 1.')
            self.profiler = None

    def _build_profiler(self):
        from lmdeploy.pytorch import envs
        activities = []
        if envs.torch_profile_cpu:
            activities.append(ProfilerActivity.CPU)
        if envs.torch_profile_cuda:
            activities.append(ProfilerActivity.CUDA)
        if len(activities) > 0:
            logger.warning(f'Profiler start on {self.name}. '
                           'Please Note that profiling might harm performance.')
            profiler = profile(activities=activities)
            return profiler
        else:
            return None

    def dump(self, merge_profiles: bool = False):
        """Dump profile result."""
        if self.profiler is None:
            return

        if not self._started:
            logger.warning(f'Profiler {self.name} not started, skip dump.')
            return

        try:
            self.profiler.stop()
            dump_path = self._get_dump_path()
            self.profiler.export_chrome_trace(dump_path)
            logger.warning(f'Profiler {self.name} dump to {dump_path}.')
        except Exception as e:
            logger.error(f'Failed to dump profile {self.name} result: {e}')
        finally:
            self.profiler = None

        if self.merge_profiles and merge_profiles:
            self._merge_profiles()

    def _merge_profiles(self):
        try:
            if self.world_size > 1:
                dist.barrier(group=self.cpu_group)
            if self.rank != 0:
                return

            merger = ProfileMerger(output_prefix=self.prefix,
                                   world_size=self.world_size,
                                   use_gzip=self.use_gzip)
            output_path, event_count = merger.merge()
            logger.warning(f'Merged {self.world_size} profile traces ({event_count} events) to {output_path}.')
        except Exception as e:
            logger.error(f'Failed to merge profile traces: {e}')

    def _prepare_profile_outputs(self):
        paths = [self._get_dump_path()]
        if self.rank == 0:
            merger = ProfileMerger(output_prefix=self.prefix,
                                   world_size=self.world_size,
                                   use_gzip=self.use_gzip)
            paths.append(merger.output_path)
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def _get_dump_path(self):
        suffix = '.json.gz' if self.use_gzip else '.json'
        return f'{self.prefix}{self.rank}{suffix}'

    async def profile_task(self):
        """Profile task."""
        if self.profiler is None:
            return

        # start profiler with delay
        await asyncio.sleep(self.delay)
        if self.merge_profiles:
            self._prepare_profile_outputs()
        self.profiler.start()
        self._started = True

        if self.duration <= 0:
            return

        # dump profiler
        await asyncio.sleep(self.duration)
        self.dump(merge_profiles=True)

    def create_task(self):
        """Create task."""
        event_loop = asyncio.get_event_loop()
        self._task = event_loop.create_task(self.profile_task())

"""
Pipeline Framework for High-Performance Data Processing.

This package provides a multi-process pipeline framework with:
- Multi-priority partitioned queues.
- Reference-counted shared memory for zero-copy data sharing.
- Decorator-based API for easy node definition.
- Real-time web-based monitoring dashboard.
"""
from .message_queue import (
    Message,
    BasePriorityQueue,
    ManagerPriorityQueue,
    PartitionedPriorityQueue,
    SharedMemoryStore,
    SharedMemoryPool,
    run_worker_loop,
    pack_payload,
    unpack_payload,
    free_payload,
    set_default_pool,
    get_default_pool,
    acquire_for_group,
    load_array_from_payload,
    _cleanup_shm,
    set_shm_registry,
    set_shm_master,
    Empty,
    Full
)

from .pipeline import Pipeline, PipelineNode, node

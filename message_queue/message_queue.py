"""
High-Performance IPC and Shared Memory Pipeline Infrastructure.

This module provides the core components for building multi-process data pipelines:
1. Message: A structured data container with priority support.
2. PartitionedPriorityQueue: A multi-partition queue for node-to-node communication.
3. SharedMemoryPool: A reference-counted pool for zero-copy NumPy array sharing.
4. run_worker_loop: The standard execution loop for pipeline nodes.

The system is designed to handle large data (like images) efficiently by 
automatically offloading NumPy arrays to shared memory.
"""
from __future__ import annotations

import multiprocessing
import threading
import time
import uuid
import pickle
import logging
import atexit
import signal
import os
import contextlib
import random
from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple, List, Set, Iterable, Union, Dict, Callable
import numpy as np
from multiprocessing import shared_memory

try:
    import redis
    _HAS_REDIS = True
except Exception:
    redis = None
    _HAS_REDIS = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# Global registry for shared memory cleanup
_SHM_POOLS: Set[SharedMemoryPool] = set()
_EPHEMERAL_SHM: Set[str] = set()
_ALL_CREATED_SHM: Set[str] = set() # Track all SHM created by this process
_SHM_CACHE: Dict[str, shared_memory.SharedMemory] = {} # Cache for pool attachments
_SHM_LOCK = threading.RLock()
_DEFAULT_POOL: Optional[SharedMemoryPool] = None
_SHARED_SHM_REGISTRY: Optional[List[str]] = None
_I_AM_SHM_MASTER: bool = False


def _get_shm_attachment(name: str) -> shared_memory.SharedMemory:
    """Get or create a cached attachment to a shared memory segment."""
    with _SHM_LOCK:
        if name in _SHM_CACHE:
            return _SHM_CACHE[name]
        
        shm = shared_memory.SharedMemory(name=name)
        # Unregister immediately to avoid resource_tracker issues
        try:
            from multiprocessing import resource_tracker
            name_to_unreg = getattr(shm, '_name', shm.name)
            resource_tracker.unregister(name_to_unreg, 'shared_memory')
        except Exception:
            pass
        _SHM_CACHE[name] = shm
        return shm


def set_default_pool(pool: SharedMemoryPool):
    global _DEFAULT_POOL
    _DEFAULT_POOL = pool


def get_default_pool() -> Optional[SharedMemoryPool]:
    return _DEFAULT_POOL


def set_shm_registry(registry: Optional[List[str]]):
    global _SHARED_SHM_REGISTRY
    _SHARED_SHM_REGISTRY = registry


def set_shm_master(is_master: bool = True):
    global _I_AM_SHM_MASTER
    _I_AM_SHM_MASTER = is_master


def _cleanup_shm():
    """Global cleanup for shared memory to prevent leaks on exit/crash."""
    # Use a flag to avoid re-entry hangs
    if getattr(_cleanup_shm, '_running', False):
        return
    _cleanup_shm._running = True
    
    try:
        with _SHM_LOCK:
            # 1. Cleanup pools (Manager-dependent, might hang)
            for pool in list(_SHM_POOLS):
                try:
                    # Use a very short timeout for pool closure during global cleanup
                    pool.close()
                except Exception:
                    pass
            _SHM_POOLS.clear()

            # 2. Cleanup all SHM created by this process (Manager-independent, safe)
            for name in list(_ALL_CREATED_SHM):
                try:
                    from multiprocessing import shared_memory
                    shm = shared_memory.SharedMemory(name=name)
                    shm.unlink()
                    shm.close()
                except Exception:
                    pass
            _ALL_CREATED_SHM.clear()
            _EPHEMERAL_SHM.clear()

            # 3. Close cached attachments
            for name, shm in list(_SHM_CACHE.items()):
                try:
                    shm.close()
                except Exception:
                    pass
            _SHM_CACHE.clear()

            # 4. Cleanup from shared registry if available (Manager-dependent)
            # ONLY the master process should do this to avoid redundant work and hangs in workers
            if _I_AM_SHM_MASTER and _SHARED_SHM_REGISTRY is not None:
                try:
                    # list() on a Manager.list() can hang if Manager is dead/stuck
                    # We try to copy it quickly. 
                    shm_names = list(_SHARED_SHM_REGISTRY)
                    for name in shm_names:
                        try:
                            from multiprocessing import shared_memory
                            shm = shared_memory.SharedMemory(name=name)
                            shm.unlink()
                            shm.close()
                        except Exception:
                            pass
                except Exception:
                    pass
    finally:
        _cleanup_shm._running = False


class Empty(Exception):
    pass


class Full(Exception):
    pass


class SharedMemoryOOMError(Exception):
    """Raised when SharedMemoryPool cannot allocate enough space."""
    pass


@dataclass
class Message:
    id: str
    priority: int
    payload: Any
    msg_type: str = "task"
    reply_to: Optional[str] = None
    timestamp: float = 0.0
    meta: dict = None

    def __init__(self, payload: Any, priority: int = 50, msg_type: str = "task", reply_to: Optional[str] = None, meta: Optional[dict] = None):
        self.id = str(uuid.uuid4())
        self.priority = int(priority)
        self.payload = payload
        self.msg_type = msg_type
        self.reply_to = reply_to
        self.timestamp = time.time()
        self.meta = meta or {}

    def serialize(self) -> bytes:
        return pickle.dumps(asdict(self))

    @staticmethod
    def deserialize(b: bytes) -> "Message":
        d = pickle.loads(b)
        m = Message(payload=d.get('payload'), priority=d.get('priority', 50), msg_type=d.get('msg_type', 'task'), reply_to=d.get('reply_to'))
        m.id = d.get('id')
        m.timestamp = d.get('timestamp', time.time())
        m.meta = d.get('meta', {})
        return m


class BasePriorityQueue:
    def put(self, message: Message) -> None:
        raise NotImplementedError()

    def get(self, block: bool = True, timeout: Optional[float] = None, msg_filter: Optional[callable] = None) -> Message:
        raise NotImplementedError()

    def peek(self, msg_filter: Optional[callable] = None) -> Optional[Message]:
        """Non-destructive peek for the first message matching msg_filter.

        Returns a Message instance or None if no matching message found.
        Implementations should not remove the message from the queue.
        """
        raise NotImplementedError()

    def put_back(self, message: Message) -> None:
        """Put a message back into the queue (best-effort)."""
        raise NotImplementedError()

    def snapshot_summary(self) -> dict:
        """Return a small diagnostic summary (counts by msg_type).

        Useful for monitoring and debugging.
        """
        raise NotImplementedError()

    def size(self) -> int:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()


class ManagerPriorityQueue(BasePriorityQueue):
    """
    A simple manager-backed multi-priority queue. Uses a Manager().list proxy
    to hold (priority, counter, serialized_message) tuples and a Manager().Lock
    to serialize access. Lower `priority` value means higher priority.

    This implementation trades raw throughput for simplicity and portability
    across processes on the same host. For cross-host usage, use
    `RedisPriorityQueue` if Redis is available.
    """

    def __init__(self, manager: Optional[Any] = None):
        self._own_manager = False
        if manager is None:
            self._manager = multiprocessing.Manager()
            self._own_manager = True
        else:
            self._manager = manager

        self._list = self._manager.list()  # holds tuples (priority, counter, ts, bytes)
        self._lock = self._manager.Lock()
        self._cond = self._manager.Condition(self._lock)
        self._counter = self._manager.Value('L', 0)
        self._closed = self._manager.Value('b', False)

    def __getstate__(self):
        state = self.__dict__.copy()
        # SyncManager objects are not picklable (contain weakrefs)
        state['_manager'] = None
        return state

    def put(self, message: Message) -> None:
        data = message.serialize()
        with self._lock:
            cnt = self._counter.value
            self._list.append((int(message.priority), int(cnt), float(message.timestamp), data))
            self._counter.value = cnt + 1
            # print(f"DEBUG: Queue put msg {message.id} type {message.msg_type}. List size: {len(self._list)}", flush=True)
            self._cond.notify_all()

    def peek(self, msg_filter: Optional[callable] = None) -> Optional[Message]:
        with self._lock:
            for item in list(self._list):
                score, cnt, ts, data = item
                try:
                    candidate = Message.deserialize(data)
                except Exception:
                    continue
                if msg_filter is None or msg_filter(candidate):
                    return candidate
        return None

    def put_back(self, message: Message) -> None:
        # best-effort: re-insert with current counter to avoid losing it
        data = message.serialize()
        with self._lock:
            cnt = self._counter.value
            self._list.append((int(message.priority), int(cnt), float(message.timestamp), data))
            self._counter.value = cnt + 1
            self._cond.notify_all()

    def snapshot_summary(self) -> dict:
        counts = {}
        with self._lock:
            for item in list(self._list):
                try:
                    msg = Message.deserialize(item[3])
                    counts[msg.msg_type] = counts.get(msg.msg_type, 0) + 1
                except Exception:
                    counts['invalid'] = counts.get('invalid', 0) + 1
        return counts

    def get(self, block: bool = True, timeout: Optional[float] = None, msg_filter: Optional[callable] = None) -> Message:
        start = time.time()
        with self._lock:
            while True:
                if self._closed.value:
                    raise Empty("Queue closed")

                if len(self._list) > 0:
                    # find smallest (priority, counter) that matches msg_filter
                    best_idx = None
                    best = None
                    for idx in range(0, len(self._list)):
                        item = self._list[idx]
                        score, cnt, ts, data = item
                        try:
                            candidate = Message.deserialize(data)
                        except Exception:
                            continue
                        if msg_filter is not None and not msg_filter(candidate):
                            # print(f"DEBUG: Queue get filtered out msg {candidate.id} type {candidate.msg_type}", flush=True)
                            continue
                        if best is None or (score, cnt) < (best[0], best[1]):
                            best = (score, cnt)
                            best_idx = idx
                    
                    if best_idx is not None:
                        tup = self._list.pop(best_idx)
                        _, _, _, data = tup
                        return Message.deserialize(data)

                if not block:
                    raise Empty()
                
                remaining = None
                if timeout is not None:
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        raise Empty()
                
                # Wait for notification or timeout
                if not self._cond.wait(timeout=remaining):
                    # If wait returned False, it timed out
                    if timeout is not None:
                        raise Empty()

    def size(self) -> int:
        with self._lock:
            return len(self._list)

    def cleanup_payloads(self, pool: Optional[SharedMemoryPool] = None) -> None:
        """Free all payloads currently in the queue."""
        # Use a timeout to avoid hanging if the lock is held by a dead process
        if self._lock.acquire(timeout=2.0):
            try:
                for item in list(self._list):
                    try:
                        # item is (priority, counter, ts, data)
                        msg = Message.deserialize(item[3])
                        free_payload(msg.payload, pool=pool)
                    except Exception:
                        pass
                self._list[:] = []
            finally:
                self._lock.release()

    def close(self) -> None:
        if self._lock.acquire(timeout=2.0):
            try:
                self._closed.value = True
            finally:
                self._lock.release()
        if self._own_manager:
            try:
                self._manager.shutdown()
            except Exception:
                pass


class RedisPriorityQueue(BasePriorityQueue):
    """
    A Redis-backed priority queue using a sorted set. Lower score = higher priority.
    The score is computed from priority and a counter to ensure FIFO among same
    priority items.
    """

    def __init__(self, redis_url: str = "redis://127.0.0.1:6379/0", key: str = "mpq:queue"):
        if not _HAS_REDIS:
            raise RuntimeError("redis package not available")
        self._r = redis.from_url(redis_url)
        self._key = key
        self._counter_key = f"{key}:counter"

    def _next_counter(self) -> int:
        return int(self._r.incr(self._counter_key))

    def put(self, message: Message) -> None:
        data = message.serialize()
        cnt = self._next_counter()
        # score: priority * 1e12 + counter (so lower priority value -> lower score)
        score = int(message.priority) * 10**12 + cnt
        self._r.zadd(self._key, {data: score})

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Message:
        start = time.time()
        while True:
            res = self._r.zrange(self._key, 0, 0)
            if res:
                data = res[0]
                # attempt to remove
                removed = self._r.zrem(self._key, data)
                if removed:
                    return Message.deserialize(data)
            if not block:
                raise Empty()
            if timeout is not None and (time.time() - start) >= timeout:
                raise Empty()
            time.sleep(0.01)

    def size(self) -> int:
        return int(self._r.zcard(self._key))

    def close(self) -> None:
        # nothing to do for redis client
        pass

    def peek(self, msg_filter: Optional[callable] = None) -> Optional[Message]:
        # best-effort: read first element and test filter, non-destructive
        res = self._r.zrange(self._key, 0, -1)
        for data in res:
            try:
                msg = Message.deserialize(data)
            except Exception:
                continue
            if msg_filter is None or msg_filter(msg):
                return msg
        return None

    def put_back(self, message: Message) -> None:
        # re-insert message with a new counter
        self.put(message)

    def snapshot_summary(self) -> dict:
        counts = {}
        res = self._r.zrange(self._key, 0, -1)
        for data in res:
            try:
                msg = Message.deserialize(data)
                counts[msg.msg_type] = counts.get(msg.msg_type, 0) + 1
            except Exception:
                counts['invalid'] = counts.get('invalid', 0) + 1
        return counts


class PartitionedPriorityQueue(BasePriorityQueue):
    """
    A multi-partition priority queue for inter-node communication.
    
    Each partition acts as an independent queue with its own capacity limit.
    This allows the pipeline to manage backpressure on a per-node basis.
    
    Attributes:
        manager: A multiprocessing.Manager instance for cross-process synchronization.
        partitions: Initial mapping of partition names to their capacities.
        default_capacity: Default capacity for new partitions.
    """

    def __init__(self, manager: Optional[Any] = None, partitions: Optional[dict] = None, default_capacity: int = 1024, stop_event: Optional[Any] = None):
        self._own_manager = False
        if manager is None:
            self._manager = multiprocessing.Manager()
            self._own_manager = True
        else:
            self._manager = manager

        self._stop_event = stop_event
        # partitions: mapping partition_name -> capacity
        self._queues = {}
        self._capacities = {}
        self._lock = self._manager.Lock()
        self._cond = self._manager.Condition(self._lock)

        if partitions:
            for name, cap in partitions.items():
                self._create_partition(name, int(cap))

        self._default_capacity = int(default_capacity)

    def __getstate__(self):
        state = self.__dict__.copy()
        # SyncManager objects are not picklable
        state['_manager'] = None
        return state

    def _create_partition(self, name: str, capacity: int):
        if name in self._queues:
            return
        q = ManagerPriorityQueue(manager=self._manager)
        self._queues[name] = q
        self._capacities[name] = int(capacity)

    def set_capacity(self, partition: str, capacity: int):
        with self._lock:
            if partition not in self._queues:
                self._create_partition(partition, capacity)
            else:
                self._capacities[partition] = int(capacity)

    def get_status(self) -> dict:
        """Return status of all partitions."""
        with self._lock:
            return {
                name: {
                    'size': q.size(),
                    'capacity': self._capacities.get(name, self._default_capacity)
                }
                for name, q in self._queues.items()
            }

    def put(self, message: Message, partition: str, block: bool = True, timeout: Optional[float] = None) -> None:
        # ensure partition exists
        if partition not in self._queues:
            with self._lock:
                if partition not in self._queues:
                    self._create_partition(partition, self._default_capacity)

        q = self._queues[partition]
        start = time.time()
        while True:
            if self._stop_event and self._stop_event.is_set():
                raise Full("Pipeline stopping")

            with self._lock:
                cap = self._capacities.get(partition, self._default_capacity)
                if q.size() < cap:
                    q.put(message)
                    self._cond.notify_all()
                    return
                
                if not block:
                    raise Full(f"partition '{partition}' is full")
                
                # Calculate wait time: use a small interval to check stop_event
                wait_timeout = 0.5
                if timeout is not None:
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        raise Full(f"timeout while waiting to put into partition '{partition}'")
                    wait_timeout = min(wait_timeout, remaining)
                
                try:
                    self._cond.wait(timeout=wait_timeout)
                except (EOFError, BrokenPipeError, Exception):
                    # Manager might be shutting down
                    raise Full("Queue manager is unavailable (shutting down)")

    def get(self, block: bool = True, timeout: Optional[float] = None, partition: Optional[str] = None, msg_filter: Optional[callable] = None) -> Message:
        start = time.time()
        
        while True:
            if self._stop_event and self._stop_event.is_set():
                raise Empty("Pipeline stopping")

            try:
                with self._lock:
                    # If partition is specified, try only that one
                    if partition is not None:
                        if partition not in self._queues:
                            # Partition doesn't exist yet
                            pass 
                        else:
                            q = self._queues[partition]
                            try:
                                msg = q.get(block=False, msg_filter=msg_filter)
                                self._cond.notify_all()
                                return msg
                            except Empty:
                                pass
                    else:
                        # Try all partitions
                        for p_name, q in list(self._queues.items()):
                            try:
                                msg = q.get(block=False, msg_filter=msg_filter)
                                self._cond.notify_all()
                                return msg
                            except Empty:
                                continue

                    if not block:
                        raise Empty()

                    # Calculate wait time: use a small interval to check stop_event
                    wait_timeout = 0.5
                    if timeout is not None:
                        remaining = timeout - (time.time() - start)
                        if remaining <= 0:
                            raise Empty()
                        wait_timeout = min(wait_timeout, remaining)
                    
                    self._cond.wait(timeout=wait_timeout)
                    if timeout is not None and (time.time() - start) >= timeout:
                        raise Empty()
            except (EOFError, BrokenPipeError, Exception) as e:
                if isinstance(e, Empty): raise
                raise Empty("Queue manager is unavailable (shutting down)")

    def peek(self, msg_filter: Optional[callable] = None, partition: Optional[str] = None) -> Optional[Message]:
        if partition is None:
            # best-effort: peek across partitions and return first match
            for q in list(self._queues.values()):
                c = q.peek(msg_filter=msg_filter)
                if c is not None:
                    return c
            return None
        if partition not in self._queues:
            return None
        return self._queues[partition].peek(msg_filter=msg_filter)

    def put_back(self, message: Message, partition: str) -> None:
        if partition not in self._queues:
            with self._lock:
                if partition not in self._queues:
                    self._create_partition(partition, self._default_capacity)
        self._queues[partition].put_back(message)

    def snapshot_summary(self) -> dict:
        summary = {}
        with self._lock:
            for name, q in self._queues.items():
                summary[name] = {'size': q.size(), 'counts': q.snapshot_summary(), 'capacity': self._capacities.get(name, self._default_capacity)}
        return summary

    def size(self, partition: Optional[str] = None) -> int:
        if partition is None:
            return sum(q.size() for q in self._queues.values())
        if partition not in self._queues:
            return 0
        return self._queues[partition].size()

    def cleanup_payloads(self, pool: Optional[SharedMemoryPool] = None) -> None:
        """Free all payloads in all partitions."""
        if self._lock.acquire(timeout=2.0):
            try:
                for q in list(self._queues.values()):
                    try:
                        q.cleanup_payloads(pool=pool)
                    except Exception:
                        pass
            finally:
                self._lock.release()

    def close(self) -> None:
        if self._lock.acquire(timeout=2.0):
            try:
                for q in list(self._queues.values()):
                    try:
                        q.close()
                    except Exception:
                        pass
                self._queues.clear()
                self._capacities.clear()
            finally:
                self._lock.release()
        if self._own_manager:
            try:
                self._manager.shutdown()
            except Exception:
                pass


class SharedMemoryStore:
    """Utility to store/retrieve numpy arrays via POSIX shared memory.

    Stores an ndarray into a SharedMemory block and returns a small meta dict
    which can be safely sent via the queue. Any process can reconstruct the
    ndarray from the meta dict. Call `unlink` when the buffer is no longer
    needed to release system resources.
    """

    @staticmethod
    def store_array(arr: np.ndarray) -> dict:
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        # write bytes
        shm.buf[:nbytes] = arr.tobytes()
        meta = {
            'name': shm.name,
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'nbytes': nbytes,
        }
        # register for cleanup
        with _SHM_LOCK:
            _EPHEMERAL_SHM.add(shm.name)
            _ALL_CREATED_SHM.add(shm.name)
        
        if _SHARED_SHM_REGISTRY is not None:
            try:
                _SHARED_SHM_REGISTRY.append(shm.name)
            except Exception:
                pass

        # Unregister from resource_tracker to avoid warnings on exit
        try:
            from multiprocessing import resource_tracker
            # Try both _name and name just in case
            name_to_unreg = getattr(shm, '_name', shm.name)
            resource_tracker.unregister(name_to_unreg, 'shared_memory')
        except Exception:
            pass
            
        # close in this process; other processes can attach by name
        shm.close()
        return meta

    @staticmethod
    def load_array(meta: dict, copy: bool = True) -> np.ndarray:
        shm_name = meta.get('shm_name') or meta.get('name')
        shm = shared_memory.SharedMemory(name=shm_name)
        
        # Unregister from resource_tracker to avoid warnings in workers
        try:
            from multiprocessing import resource_tracker
            name_to_unreg = getattr(shm, '_name', shm.name)
            resource_tracker.unregister(name_to_unreg, 'shared_memory')
        except Exception:
            pass

        offset = meta.get('offset', 0)
        nbytes = meta.get('nbytes')
        
        if nbytes is None:
            # Fallback for old meta or ephemeral without nbytes
            arr = np.ndarray(meta['shape'], dtype=np.dtype(meta['dtype']), buffer=shm.buf)
        else:
            arr = np.ndarray(meta['shape'], dtype=np.dtype(meta['dtype']), buffer=shm.buf[offset:offset+nbytes])
            
        if copy:
            out = arr.copy()
            shm.close()
            return out
        else:
            # caller must call shm.close() eventually; we keep shm open reference
            return arr

    @staticmethod
    def unlink(name: str) -> None:
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.unlink()
            shm.close()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        finally:
            with _SHM_LOCK:
                if name in _EPHEMERAL_SHM:
                    _EPHEMERAL_SHM.remove(name)
                if name in _ALL_CREATED_SHM:
                    _ALL_CREATED_SHM.remove(name)


class SharedMemoryPool:
    """
    A reference-counted pool for managing Shared Memory blocks with fragmented allocation.
    
    This pool pre-allocates a large shared memory segment and manages it in fragments.
    It supports reference counting, allowing multiple messages to point to the same data block.
    
    When a block's reference count reaches zero, it is returned to the free list and merged.
    If an array is larger than the total pool size, it falls back to ephemeral allocation.
    """

    def __init__(self, free_ranges, allocations, ref_counts, lock, condition, total_size: int, prefix: str = 'shmpool', stop_event: Optional[Any] = None):
        self._free_ranges = free_ranges # manager.list of (shm_name, offset, size)
        self._allocations = allocations # manager.dict of alloc_id -> (shm_name, offset, size)
        self._ref_counts = ref_counts
        self._lock = lock
        self._condition = condition
        self._stop_event = stop_event
        self.total_size = total_size
        self.prefix = prefix
        self._shm_names = set() # Track all segments used by this pool

    @classmethod
    def create(cls, manager: Any, pool_size: int = 8, block_size: int = 1024 * 1024, prefix: str = 'shmpool', stop_event: Optional[Any] = None) -> 'SharedMemoryPool':
        total_size = pool_size * block_size
        shm = shared_memory.SharedMemory(create=True, size=total_size)
        shm_name = shm.name
        
        # In Python 3.8+, we need to unregister from resource_tracker 
        try:
            from multiprocessing import resource_tracker
            name_to_unreg = getattr(shm, '_name', shm.name)
            resource_tracker.unregister(name_to_unreg, 'shared_memory')
        except Exception:
            pass
        shm.close()

        free_ranges = manager.list([(shm_name, 0, total_size)])
        allocations = manager.dict()
        ref_counts = manager.dict()
        lock = manager.Lock()
        condition = manager.Condition(lock)
        
        pool = cls(free_ranges, allocations, ref_counts, lock, condition, total_size, prefix=prefix, stop_event=stop_event)
        pool._shm_names.add(shm_name)
        
        # register for cleanup
        with _SHM_LOCK:
            _ALL_CREATED_SHM.add(shm_name)
            _SHM_POOLS.add(pool)
            
        if _SHARED_SHM_REGISTRY is not None:
            try:
                _SHARED_SHM_REGISTRY.append(shm_name)
            except Exception:
                pass
        return pool

    def store_array(self, arr: np.ndarray, timeout: Optional[float] = None) -> dict:
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        
        if nbytes > self.total_size:
            raise SharedMemoryOOMError(
                f"Requested {nbytes} bytes, but total pool size is only {self.total_size} bytes. "
                "This allocation can never succeed in the current pool."
            )

        start_time = time.time()
        alloc_info = None # (shm_name, offset, nbytes, alloc_id)
        
        with self._condition:
            while True:
                if self._stop_event and self._stop_event.is_set():
                    raise SharedMemoryOOMError("Pipeline stopping")

                # Find first-fit
                best_idx = -1
                for i, (shm_name, offset, size) in enumerate(self._free_ranges):
                    if size >= nbytes:
                        best_idx = i
                        break
                
                if best_idx != -1:
                    shm_name, offset, size = self._free_ranges.pop(best_idx)
                    if size > nbytes:
                        # Put the remainder back
                        self._free_ranges.append((shm_name, offset + nbytes, size - nbytes))
                    
                    alloc_id = f"pool_{shm_name}_{offset}_{uuid.uuid4().hex[:8]}"
                    self._allocations[alloc_id] = (shm_name, offset, nbytes)
                    self._ref_counts[alloc_id] = 1
                    alloc_info = (shm_name, offset, nbytes, alloc_id)
                    break
                
                # No space, wait
                wait_timeout = 0.5
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        raise SharedMemoryOOMError(
                            f"Timed out after {timeout}s waiting for {nbytes} bytes in SharedMemoryPool. "
                            "Pool is likely fragmented or leaked."
                        )
                    wait_timeout = min(wait_timeout, remaining)
                
                self._condition.wait(timeout=wait_timeout)
                if timeout is not None and (time.time() - start_time) >= timeout:
                    raise SharedMemoryOOMError(f"Timed out waiting for {nbytes} bytes in SharedMemoryPool.")

        shm_name, offset, nbytes, alloc_id = alloc_info
        # write into block using cached attachment
        shm = _get_shm_attachment(shm_name)
        shm.buf[offset:offset+nbytes] = arr.tobytes()
        
        meta = {
            'name': alloc_id, 
            'shm_name': shm_name,
            'offset': offset,
            'nbytes': nbytes,
            'shape': arr.shape, 
            'dtype': str(arr.dtype), 
            'pool': True
        }
        return meta

    def load_array(self, meta: dict, copy: bool = True) -> np.ndarray:
        """Load an array from the pool using metadata."""
        if not meta.get('pool'):
            return SharedMemoryStore.load_array(meta, copy=copy)
            
        shm_name = meta.get('shm_name') or meta.get('name')
        offset = meta.get('offset', 0)
        nbytes = meta.get('nbytes')
        
        shm = _get_shm_attachment(shm_name)
            
        arr = np.ndarray(meta['shape'], dtype=np.dtype(meta['dtype']), 
                         buffer=shm.buf[offset:offset+nbytes])
        if copy:
            return arr.copy()
        else:
            return arr

    @contextlib.contextmanager
    def acquire_array(self, meta: dict, copy: bool = False):
        """
        Context manager to load an array and automatically free it when done.
        
        This implements the 'manage memory release and return' pattern.
        """
        arr = self.load_array(meta, copy=copy)
        try:
            yield arr
        finally:
            self.free(meta)

    def inc_ref(self, name: str, delta: int = 1) -> None:
        """Increment reference count for a block."""
        with self._condition:
            if name in self._ref_counts:
                self._ref_counts[name] += delta
            else:
                # If it's not in ref_counts, it might be ephemeral or already freed
                pass

    def free(self, meta_or_id: Union[str, dict]) -> None:
        """Decrement reference count; only actually free if it hits 0."""
        if isinstance(meta_or_id, dict):
            alloc_id = meta_or_id.get('name')
        else:
            alloc_id = meta_or_id
            
        if not alloc_id:
            return

        with self._condition:
            if alloc_id not in self._ref_counts:
                # If lock failed or not in pool, try ephemeral unlink if it's a dict
                if isinstance(meta_or_id, dict) and not meta_or_id.get('pool'):
                    SharedMemoryStore.unlink(meta_or_id['name'])
                return
                
            count = self._ref_counts[alloc_id] - 1
            if count > 0:
                self._ref_counts[alloc_id] = count
                return
                
            # count <= 0, cleanup
            if alloc_id in self._ref_counts:
                del self._ref_counts[alloc_id]

            if alloc_id in self._allocations:
                shm_name, offset, nbytes = self._allocations.pop(alloc_id)
                self._free_ranges.append((shm_name, offset, nbytes))
                self._merge_ranges()
                self._condition.notify_all()

    def _merge_ranges(self):
        """Merge adjacent free ranges to reduce fragmentation."""
        if not self._free_ranges:
            return
        # Group by shm_name
        by_shm = {}
        for shm_name, offset, size in self._free_ranges:
            by_shm.setdefault(shm_name, []).append((offset, size))
        
        new_ranges = []
        for shm_name, ranges in by_shm.items():
            ranges.sort()
            if not ranges: continue
            curr_off, curr_size = ranges[0]
            for next_off, next_size in ranges[1:]:
                if curr_off + curr_size == next_off:
                    curr_size += next_size
                else:
                    new_ranges.append((shm_name, curr_off, curr_size))
                    curr_off, curr_size = next_off, next_size
            new_ranges.append((shm_name, curr_off, curr_size))
        
        # Update the manager list in-place
        del self._free_ranges[:]
        self._free_ranges.extend(new_ranges)

    def close(self) -> None:
        """Unlink all allocated blocks and the pool itself."""
        with self._condition:
            for shm_name in list(self._shm_names):
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.unlink()
                    shm.close()
                except Exception:
                    pass
                with _SHM_LOCK:
                    if shm_name in _ALL_CREATED_SHM:
                        _ALL_CREATED_SHM.remove(shm_name)
            
            self._shm_names.clear()
            del self._free_ranges[:]
            self._allocations.clear()
            self._ref_counts.clear()
            self._condition.notify_all()
        
        with _SHM_LOCK:
            if self in _SHM_POOLS:
                _SHM_POOLS.remove(self)

    def get_status(self) -> dict:
        """Return pool usage statistics."""
        with self._condition:
            total_free = sum(size for _, _, size in self._free_ranges)
            return {
                'total_size': self.total_size,
                'free_size': total_free,
                'used_size': self.total_size - total_free,
                'num_allocations': len(self._allocations),
                'ref_counts': dict(self._ref_counts)
            }


class QueueMonitor:
    """Lightweight monitor for a ManagerPriorityQueue.

    Starts a background thread that periodically logs a snapshot summary.
    Useful in production to detect stuck queues or unexpected message types.
    """

    def __init__(self, queue: BasePriorityQueue, interval: float = 5.0, logger_name: str = None):
        self.queue = queue
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._logger = logging.getLogger(logger_name or __name__)

    def start(self):
        if not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                summary = self.queue.snapshot_summary()
                self._logger.info(f"QueueMonitor snapshot: {summary}")
            except Exception as e:
                self._logger.exception("QueueMonitor error: %s", e)
            self._stop.wait(self.interval)


# Helper utilities for consumer simplification
def acquire_for_group(queue: BasePriorityQueue, desired_types, group: Optional[str] = None, timeout: Optional[float] = None, poll_interval: float = 0.01, partition: Optional[str] = None, msg_filter: Optional[callable] = None) -> Message:
    """Acquire a message atomically for a consumer group.

    This helper uses the queue's native `get` with a filter to ensure
    atomicity and fair competition among workers.
    """
    desired_set = set(desired_types)
    
    def combined_filter(m: Message) -> bool:
        # 1. Match desired types
        type_match = False
        if m.msg_type in desired_set:
            type_match = True
        # Match shutdown targeted to this group (or all)
        elif m.msg_type == 'shutdown':
            if group is None:
                type_match = True
            elif isinstance(m.payload, dict):
                target = m.payload.get('target')
                if target in (None, group):
                    type_match = True
        
        if not type_match:
            return False
            
        # 2. Apply external filter if provided
        if msg_filter is not None:
            return msg_filter(m)
            
        return True

    return _get(queue, block=True, timeout=timeout, msg_filter=combined_filter, partition=partition)


def load_array_from_payload(payload: dict, copy: bool = True, pool: Optional['SharedMemoryPool'] = None) -> Optional[np.ndarray]:
    """Load an ndarray from a message payload by recursively searching for SHM metas.

    This helper is now fully symmetric with pack_payload: it scans the entire
    payload and returns the FIRST valid ndarray it finds (either direct or in SHM).
    """
    if pool is None:
        pool = get_default_pool()

    # Accept Message instances
    if isinstance(payload, Message):
        payload = payload.payload

    if isinstance(payload, np.ndarray):
        return payload.copy() if copy else payload

    if not isinstance(payload, dict):
        return None

    def _find_and_load(obj):
        if isinstance(obj, np.ndarray):
            return obj.copy() if copy else obj
        
        if isinstance(obj, dict):
            # Check if this dict is a SHM meta
            if obj.get('name') and (obj.get('shape') or obj.get('dtype') or 'pool' in obj):
                if pool and obj.get('pool'):
                    return pool.load_array(obj, copy=copy)
                try:
                    return SharedMemoryStore.load_array(obj, copy=copy)
                except Exception:
                    pass
            
            # Recurse into values
            for v in obj.values():
                res = _find_and_load(v)
                if res is not None:
                    return res
        
        if isinstance(obj, (list, tuple)):
            for v in obj:
                res = _find_and_load(v)
                if res is not None:
                    return res
        return None

    return _find_and_load(payload)


# Partition-aware wrappers for queue operations. These attempt to call the
# partitioned signatures where available, falling back to non-partitioned
# ManagerPriorityQueue semantics.
def _peek(queue: BasePriorityQueue, msg_filter: Optional[callable] = None, partition: Optional[str] = None) -> Optional[Message]:
    try:
        # try partition-aware peek
        return queue.peek(msg_filter=msg_filter, partition=partition) if partition is not None else queue.peek(msg_filter=msg_filter)
    except TypeError:
        # fallback to legacy signature
        return queue.peek(msg_filter=msg_filter)


def _get(queue: BasePriorityQueue, block: bool = True, timeout: Optional[float] = None, msg_filter: Optional[callable] = None, partition: Optional[str] = None) -> Message:
    try:
        if partition is not None:
            return queue.get(block=block, timeout=timeout, msg_filter=msg_filter, partition=partition)
        return queue.get(block=block, timeout=timeout, msg_filter=msg_filter)
    except TypeError:
        # fallback: queue does not accept partition or msg_filter
        # try without partition first
        try:
            return queue.get(block=block, timeout=timeout, msg_filter=msg_filter)
        except TypeError:
            # try without msg_filter
            msg = queue.get(block=block, timeout=timeout)
            if msg_filter is not None and not msg_filter(msg):
                # If we got a message but it doesn't match, we have a problem with non-filtering queues.
                # For Redis, we'd need a more complex implementation.
                # For now, just return it or handle it.
                pass
            return msg


def _put(queue: BasePriorityQueue, message: Message, partition: Optional[str] = None, block: bool = True, timeout: Optional[float] = None) -> None:
    try:
        if partition is not None:
            return queue.put(message, partition=partition, block=block, timeout=timeout)
        return queue.put(message)
    except TypeError:
        # fallback: queue does not accept partition arg
        return queue.put(message)


def pack_payload(obj: Any, pool: Optional['SharedMemoryPool'] = None) -> Any:
    """Recursively pack numpy arrays in `obj` into shared memory via `pool`.

    Returns a new object (may reuse immutable values). If `pool` is None,
    it attempts to use the global default pool.
    """
    if pool is None:
        pool = get_default_pool()

    try:
        if isinstance(obj, np.ndarray):
            if pool:
                return pool.store_array(obj)
            else:
                logger.warning("No SharedMemoryPool provided for pack_payload. Falling back to ephemeral allocation.")
                return SharedMemoryStore.store_array(obj)
        
        if isinstance(obj, dict):
            # detect if this dict *looks like* a shm meta
            if obj.get('name') and (obj.get('shape') or obj.get('dtype') or obj.get('nbytes') or 'pool' in obj):
                # If it's already a SHM meta, increment ref count if using a pool
                if pool and obj.get('pool'):
                    pool.inc_ref(obj['name'])
                return obj
            
            out = {}
            for k, v in obj.items():
                out[k] = pack_payload(v, pool)
            return out
        
        if isinstance(obj, list):
            return [pack_payload(v, pool) for v in obj]
        if isinstance(obj, tuple):
            return tuple(pack_payload(v, pool) for v in obj)
        return obj
    except SharedMemoryOOMError:
        raise
    except Exception:
        return obj


def unpack_payload(payload: Any, pool: Optional['SharedMemoryPool'] = None, copy: bool = True) -> Any:
    """
    Recursively scan a payload and replace all Shared Memory metadata with actual arrays.
    
    This is the inverse of pack_payload. It ensures that the business logic
    receives the original data format (e.g. numpy arrays) instead of SHM metadata.
    """
    # Accept Message instances
    if isinstance(payload, Message):
        payload = payload.payload

    if isinstance(payload, dict):
        # Check if this dict is a SHM meta
        if payload.get('name') and (payload.get('shape') or payload.get('dtype') or 'pool' in payload):
            try:
                if pool is not None:
                    return pool.load_array(payload, copy=copy)
                else:
                    return SharedMemoryStore.load_array(payload, copy=copy)
            except Exception:
                return payload
        
        return {k: unpack_payload(v, pool, copy=copy) for k, v in payload.items()}
    
    if isinstance(payload, list):
        return [unpack_payload(v, pool, copy=copy) for v in payload]
    
    if isinstance(payload, tuple):
        return tuple(unpack_payload(v, pool, copy=copy) for v in payload)
    
    if isinstance(payload, np.ndarray) and copy:
        return payload.copy()
        
    return payload


def free_payload(payload: Any, pool: Optional['SharedMemoryPool'] = None) -> None:
    """
    Recursively scan a payload and free all Shared Memory blocks.
    
    This function finds all SHM metadata dicts in the payload and decrements
    their reference counts in the pool (or unlinks them if they are ephemeral).
    
    Args:
        payload: The payload to clean up (can be Message, dict, list, etc.).
        pool: The SharedMemoryPool to use for freeing.
    """
    if pool is None:
        pool = get_default_pool()

    # Accept Message instances
    if isinstance(payload, Message):
        payload = payload.payload

    def _free_meta(meta):
        if not isinstance(meta, dict):
            return
        try:
            if pool is not None:
                pool.free(meta)
            else:
                name = meta.get('name') or meta.get('shm_name')
                if name:
                    SharedMemoryStore.unlink(name)
        except Exception:
            pass

    def _scan_and_free(obj):
        if isinstance(obj, dict):
            # Check if this dict is a SHM metadata descriptor
            if obj.get('name') and (obj.get('shape') or obj.get('dtype') or 'pool' in obj):
                _free_meta(obj)
            
            for v in obj.values():
                _scan_and_free(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _scan_and_free(v)

    try:
        _scan_and_free(payload)
    except Exception:
        pass


def run_worker_loop(queue: BasePriorityQueue, business_fn, desired_types, group: Optional[str] = None, pool: Optional['SharedMemoryPool'] = None, worker_name: Optional[str] = None, processing_delay: float = 0.0, partition: Optional[str] = None, result_partition: Optional[str] = None, result_msg_type: str = 'result', on_task_start: Optional[callable] = None, on_task_done: Optional[callable] = None, on_produce: Optional[callable] = None, stop_event: Optional[Any] = None, enable_metrics: bool = True, metrics_sample_rate: float = 1.0, metrics_sample_every: int = 1, msg_filter: Optional[callable] = None):
    """
    Standard execution loop for a pipeline node worker.
    
    This loop handles:
    1. Fetching messages from the queue (with partition support).
    2. Automatically restoring NumPy arrays from Shared Memory.
    3. Executing the user-provided `business_fn`.
    4. Packing the result back into Shared Memory.
    5. Forwarding the result to the next node.
    6. Automatic cleanup of input Shared Memory blocks (reference counting).
    
    Args:
        queue: The priority queue to fetch from.
        business_fn: Callable(array, payload, message) -> result_payload.
        desired_types: List of message types this worker handles.
        group: Optional consumer group name.
        pool: SharedMemoryPool for SHM management.
        worker_name: Name for logging.
        processing_delay: Artificial delay for testing.
        partition: Input partition name.
        result_partition: Output partition name.
        result_msg_type: Type for the output message.
        on_task_start: Callback when a task is acquired.
        on_task_done: Callback when a task is finished.
        on_produce: Callback when an item is produced (sent to next stage).
        stop_event: Event to signal graceful shutdown.
        msg_filter: Optional filter for message acquisition.
    """
    if pool is None:
        pool = get_default_pool()

    name = worker_name or f"worker-{os.getpid()}"
    loop_iter = 0
    logger.info(f"{name} started")
    try:
        while not (stop_event and stop_event.is_set()):
            loop_iter += 1
            sample_hit = enable_metrics and (
                (metrics_sample_rate >= 1.0 or random.random() < metrics_sample_rate) and
                (metrics_sample_every <= 1 or (loop_iter % metrics_sample_every) == 0)
            )

            ts0 = time.perf_counter_ns() if sample_hit else None
            try:
                msg = acquire_for_group(queue, desired_types, group=group, timeout=1.0, partition=partition, msg_filter=msg_filter)
            except Empty:
                continue

            metrics = None
            if sample_hit:
                # Choose where to store timing metadata so on_task_done can consume it.
                if isinstance(msg.payload, dict):
                    metrics = msg.payload
                else:
                    if msg.meta is None:
                        msg.meta = {}
                    metrics = msg.meta
                ts1 = time.perf_counter_ns()
                metrics.setdefault('_start_time', time.time())
                metrics['_t_get'] = (ts1 - ts0) / 1e9
                metrics.setdefault('_t_put', 0.0)
                metrics.setdefault('_t_fn', 0.0)

            if msg.msg_type == 'shutdown':
                logger.info(f"{name} shutting down")
                break

            if on_task_start:
                try:
                    on_start_res = on_task_start(msg)
                except Exception:
                    pass

            # Unpack the entire payload so it contains original objects (like numpy arrays)
            # instead of SHM metadata.
            unpacked_payload = unpack_payload(msg.payload, pool=pool, copy=True)
            
            arr = None
            try:
                # For backward compatibility, still try to find the "main" array if it exists.
                # We use copy=False because unpacked_payload already contains copies.
                arr = load_array_from_payload(unpacked_payload, copy=False, pool=pool)

                # call business logic
                ts2 = time.perf_counter_ns() if sample_hit else None
                try:
                    result = business_fn(arr, unpacked_payload, msg)
                except Exception:
                    logger.exception(f"{name} business_fn error")
                    result = None
                finally:
                    if sample_hit:
                        ts3 = time.perf_counter_ns()
                        metrics['_t_fn'] = (ts3 - ts2) / 1e9
                        ts_fn_end = ts3
                    else:
                        ts_fn_end = None

                if processing_delay:
                    time.sleep(processing_delay)

                if result is not None:
                    # Handle generators for one-to-many mapping
                    if hasattr(result, "__next__") and hasattr(result, "__iter__"):
                        try:
                            item_count = 0
                            for item in result:
                                if item is None:
                                    continue
                                packed_payload = pack_payload(item, pool)
                                res_msg = Message(payload=packed_payload, priority=50, msg_type=result_msg_type)
                                _put(queue, res_msg, partition=result_partition)
                                if sample_hit and item_count == 0:
                                    ts4 = time.perf_counter_ns()
                                    metrics['_t_put'] = (ts4 - (ts_fn_end or ts1)) / 1e9
                                    metrics['_total_t'] = metrics.get('_t_get', 0.0) + metrics.get('_t_fn', 0.0) + metrics.get('_t_put', 0.0)
                                if on_produce:
                                    try:
                                        on_produce(item, msg, item_count)
                                    except Exception:
                                        pass
                                item_count += 1
                            
                            if item_count > 0:
                                msg.payload['_generator_item_count'] = item_count
                        except Exception:
                            logger.exception(f"{name} generator error")
                    else:
                        # delegate packing to the message-queue layer
                        packed_payload = pack_payload(result, pool)
                        res_msg = Message(payload=packed_payload, priority=50, msg_type=result_msg_type)
                        try:
                            _put(queue, res_msg, partition=result_partition)
                            if sample_hit:
                                ts4 = time.perf_counter_ns()
                                metrics['_t_put'] = (ts4 - (ts_fn_end or ts1)) / 1e9
                                metrics['_total_t'] = metrics.get('_t_get', 0.0) + metrics.get('_t_fn', 0.0) + metrics.get('_t_put', 0.0)
                            if on_produce:
                                try:
                                    on_produce(result, msg, 1)
                                except Exception:
                                    pass
                        except Exception:
                            # on failure, attempt fallback: send original result inline
                            res_msg = Message(payload=result, priority=50, msg_type=result_msg_type)
                            _put(queue, res_msg, partition=result_partition)
                            if sample_hit:
                                ts4 = time.perf_counter_ns()
                                metrics['_t_put'] = (ts4 - (ts_fn_end or ts1)) / 1e9
                                metrics['_total_t'] = metrics.get('_t_get', 0.0) + metrics.get('_t_fn', 0.0) + metrics.get('_t_put', 0.0)
                            if on_produce:
                                try:
                                    on_produce(result, msg, 1)
                                except Exception:
                                    pass
                else:
                    # No output produced; still record totals for visibility
                    if sample_hit:
                        metrics['_t_put'] = metrics.get('_t_put', 0.0)
                        metrics['_total_t'] = metrics.get('_t_get', 0.0) + metrics.get('_t_fn', 0.0) + metrics.get('_t_put', 0.0)
            finally:
                # free input buffer even if business_fn or packing fails
                free_payload(msg.payload, pool=pool)
                if on_task_done:
                    try:
                        on_task_done(msg)
                    except Exception:
                        pass
    except KeyboardInterrupt:
        logger.info(f"{name} interrupted")
    except Exception as e:
        logger.error(f"{name} loop error: {e}", exc_info=True)


def worker_main(queue_proxy: ManagerPriorityQueue, results_list_proxy, worker_id: int, cascade_levels: int = 2, processing_delay: float = 0.2):
    """Worker process: fetch tasks, simulate processing, optionally requeue
    for cascade, append final results to results_list_proxy.
    """
    proc_name = f"worker-{worker_id}"
    logger.info(f"{proc_name} started")
    try:
        while True:
            try:
                msg = queue_proxy.get(block=True, timeout=1.0, msg_filter=lambda m: m.msg_type in ('task', 'shutdown'))
            except Empty:
                # check again
                continue

            if msg.msg_type == 'shutdown':
                logger.info(f"{proc_name} received shutdown")
                break

            # Simulate processing
            payload = msg.payload
            # Expect payload to be dict with 'level' and 'data'
            level = payload.get('level', 0) if isinstance(payload, dict) else 0
            logger.info(f"{proc_name} processing task {msg.id} level={level} priority={msg.priority}")
            time.sleep(processing_delay)

            if level < cascade_levels - 1:
                # requeue with incremented level and slightly lower priority
                new_payload = {'level': level + 1, 'data': payload.get('data') if isinstance(payload, dict) else payload}
                new_msg = Message(payload=new_payload, priority=max(0, msg.priority - 1), msg_type='task')
                queue_proxy.put(new_msg)
                logger.info(f"{proc_name} requeued {new_msg.id} for level {level+1}")
            else:
                # produce a result
                result = {'task_id': msg.id, 'worker': worker_id, 'result': f"processed at level {level}"}
                results_list_proxy.append(result)
                logger.info(f"{proc_name} finished {msg.id} -> result appended")

    except KeyboardInterrupt:
        logger.info(f"{proc_name} interrupted")


def demo(num_workers: int = 3, num_tasks: int = 10, cascade_levels: int = 3):
    """Run a demo: master creates manager, queue, results list, spawns workers,
    dispatches tasks, and collects results.
    """
    mp = multiprocessing.Manager()
    queue = ManagerPriorityQueue(manager=mp)
    results = mp.list()

    workers: List[multiprocessing.Process] = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=worker_main, args=(queue, results, i, cascade_levels, 0.1), daemon=True)
        p.start()
        workers.append(p)

    # Dispatch tasks from a thread
    def dispatch():
        for i in range(num_tasks):
            payload = {'level': 0, 'data': f"task-{i}"}
            # randomize priority for demo
            prio = max(0, 50 - i % 10)
            msg = Message(payload=payload, priority=prio)
            queue.put(msg)
            logger.info(f"master dispatched {msg.id} priority={msg.priority}")
            time.sleep(0.02)

    dispatch_thread = threading.Thread(target=dispatch, daemon=True)
    dispatch_thread.start()

    # Collect results in main thread
    collected = []
    start = time.time()
    try:
        while len(collected) < num_tasks:
            while len(results) > 0:
                collected.append(results.pop(0))
                logger.info(f"master collected result {len(collected)}/{num_tasks}")
            time.sleep(0.05)
            # safety timeout
            if time.time() - start > 60:
                break

    finally:
        # send shutdown to workers
        for _ in workers:
            queue.put(Message(payload={}, priority=1000, msg_type='shutdown'))

        for p in workers:
            p.join(timeout=2.0)

    logger.info("Demo finished. Collected results:")
    for r in collected:
        logger.info(r)


# Register cleanup
atexit.register(_cleanup_shm)

def _signal_handler(signum, frame):
    _cleanup_shm()
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    os._exit(1)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    demo(num_workers=4, num_tasks=8, cascade_levels=3)

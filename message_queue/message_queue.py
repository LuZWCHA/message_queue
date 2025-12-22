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
_SHM_LOCK = threading.RLock()
_DEFAULT_POOL: Optional[SharedMemoryPool] = None


def set_default_pool(pool: SharedMemoryPool):
    global _DEFAULT_POOL
    _DEFAULT_POOL = pool


def get_default_pool() -> Optional[SharedMemoryPool]:
    return _DEFAULT_POOL


def _cleanup_shm():
    """Global cleanup for shared memory to prevent leaks on exit/crash."""
    with _SHM_LOCK:
        # 1. Cleanup pools (Manager-dependent, might hang)
        for pool in list(_SHM_POOLS):
            try:
                pool.close()
            except Exception:
                pass
        _SHM_POOLS.clear()

        # 2. Cleanup all SHM created by this process (Manager-independent, safe)
        for name in list(_ALL_CREATED_SHM):
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.unlink()
                shm.close()
            except Exception:
                pass
        _ALL_CREATED_SHM.clear()
        _EPHEMERAL_SHM.clear()


def _signal_handler(signum, frame):
    # logger.info(f"Received signal {signum}, cleaning up shared memory...")
    _cleanup_shm()
    # Re-raise or exit
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    exit(1)


class Empty(Exception):
    pass


class Full(Exception):
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

    def close(self) -> None:
        with self._lock:
            self._closed.value = True
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

    def __init__(self, manager: Optional[Any] = None, partitions: Optional[dict] = None, default_capacity: int = 1024):
        self._own_manager = False
        if manager is None:
            self._manager = multiprocessing.Manager()
            self._own_manager = True
        else:
            self._manager = manager

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
            with self._lock:
                cap = self._capacities.get(partition, self._default_capacity)
                if q.size() < cap:
                    q.put(message)
                    self._cond.notify_all()
                    return
                
                if not block:
                    raise Full(f"partition '{partition}' is full")
                
                remaining = None
                if timeout is not None:
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        raise Full(f"timeout while waiting to put into partition '{partition}'")
                
                try:
                    if not self._cond.wait(timeout=remaining):
                        if timeout is not None:
                            raise Full(f"timeout while waiting to put into partition '{partition}'")
                except (EOFError, BrokenPipeError, Exception):
                    # Manager might be shutting down
                    raise Full("Queue manager is unavailable (shutting down)")

    def get(self, block: bool = True, timeout: Optional[float] = None, partition: Optional[str] = None, msg_filter: Optional[callable] = None) -> Message:
        start = time.time()
        
        while True:
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

                    remaining = None
                    if timeout is not None:
                        remaining = timeout - (time.time() - start)
                        if remaining <= 0:
                            raise Empty()
                    
                    if not self._cond.wait(timeout=remaining):
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

    def close(self) -> None:
        with self._lock:
            for q in self._queues.values():
                try:
                    q.close()
                except Exception:
                    pass
            self._queues.clear()
            self._capacities.clear()
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
        
        # Unregister from resource_tracker to avoid warnings on exit
        try:
            from multiprocessing.resource_tracker import unregister
            unregister(shm._name, 'shared_memory')
        except (ImportError, AttributeError):
            pass
            
        # close in this process; other processes can attach by name
        shm.close()
        return meta

    @staticmethod
    def load_array(meta: dict, copy: bool = True) -> np.ndarray:
        shm = shared_memory.SharedMemory(name=meta['name'])
        
        # Unregister from resource_tracker to avoid warnings in workers
        try:
            from multiprocessing.resource_tracker import unregister
            unregister(shm._name, 'shared_memory')
        except (ImportError, AttributeError):
            pass

        arr = np.ndarray(meta['shape'], dtype=np.dtype(meta['dtype']), buffer=shm.buf)
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
    A reference-counted pool for managing Shared Memory blocks.
    
    This pool pre-allocates fixed-size blocks to reduce the overhead of 
    creating/destroying shared memory segments. It also supports reference 
    counting, allowing multiple messages to point to the same data block.
    
    When a block's reference count reaches zero, it is returned to the free list.
    If an array is larger than the block size, it falls back to ephemeral 
    allocation (one-time use).
    """

    def __init__(self, free_list, alloc_list, ref_counts, lock, block_size: int, prefix: str = 'shmpool'):
        self._free = free_list
        self._alloc = alloc_list
        self._ref_counts = ref_counts
        self._lock = lock
        self.block_size = int(block_size)
        self.prefix = prefix

    @classmethod
    def create(cls, manager: Any, pool_size: int = 8, block_size: int = 1024 * 1024, prefix: str = 'shmpool') -> 'SharedMemoryPool':
        free_list = manager.list()
        alloc_list = manager.list()
        ref_counts = manager.dict()
        lock = manager.Lock()
        pool = cls(free_list, alloc_list, ref_counts, lock, block_size, prefix=prefix)
        # preallocate
        for _ in range(pool_size):
            shm = shared_memory.SharedMemory(create=True, size=block_size)
            name = shm.name
            # In Python 3.8+, we need to unregister from resource_tracker 
            # if we want to manage the lifecycle ourselves across processes.
            try:
                from multiprocessing.resource_tracker import unregister
                unregister(shm._name, 'shared_memory')
            except (ImportError, AttributeError):
                pass
            shm.close()
            alloc_list.append(name)
            free_list.append(name)
            with _SHM_LOCK:
                _ALL_CREATED_SHM.add(name)
        # register for cleanup
        with _SHM_LOCK:
            _SHM_POOLS.add(pool)
        return pool

    def store_array(self, arr: np.ndarray) -> dict:
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        if nbytes > self.block_size:
            # fallback to ephemeral allocation
            meta = SharedMemoryStore.store_array(arr)
            meta['pool'] = False
            # Use timeout for lock to avoid deadlocks during shutdown
            if self._lock.acquire(timeout=2.0):
                try:
                    self._ref_counts[meta['name']] = 1
                finally:
                    self._lock.release()
            else:
                logger.warning(f"Failed to acquire pool lock for ephemeral array {meta['name']}")
                self._ref_counts[meta['name']] = 1 # Best effort
            return meta

        name = None
        if self._lock.acquire(timeout=2.0):
            try:
                if len(self._free) > 0:
                    name = self._free.pop(0)
                else:
                    # allocate a new block and track it
                    shm = shared_memory.SharedMemory(create=True, size=self.block_size)
                    name = shm.name
                    try:
                        from multiprocessing.resource_tracker import unregister
                        unregister(shm._name, 'shared_memory')
                    except (ImportError, AttributeError):
                        pass
                    shm.close()
                    self._alloc.append(name)
                    with _SHM_LOCK:
                        _ALL_CREATED_SHM.add(name)
                self._ref_counts[name] = 1
            finally:
                self._lock.release()
        else:
            logger.error("Failed to acquire pool lock for store_array")
            return SharedMemoryStore.store_array(arr) # Fallback to ephemeral

        # write into block
        shm = shared_memory.SharedMemory(name=name)
        # Unregister here too
        try:
            from multiprocessing.resource_tracker import unregister
            unregister(shm._name, 'shared_memory')
        except (ImportError, AttributeError):
            pass
        shm.buf[:nbytes] = arr.tobytes()
        shm.close()
        meta = {'name': name, 'shape': arr.shape, 'dtype': str(arr.dtype), 'nbytes': nbytes, 'pool': True, 'block_size': self.block_size}
        return meta

    def inc_ref(self, name: str, delta: int = 1) -> None:
        """Increment reference count for a block."""
        if self._lock.acquire(timeout=1.0):
            try:
                if name in self._ref_counts:
                    self._ref_counts[name] += delta
                else:
                    self._ref_counts[name] = 1 + delta
            finally:
                self._lock.release()

    def free(self, name: str) -> None:
        # Decrement ref count; only actually free if it hits 0
        if self._lock.acquire(timeout=2.0):
            try:
                count = self._ref_counts.get(name, 1) - 1
                if count > 0:
                    self._ref_counts[name] = count
                    return
                # count <= 0, cleanup
                if name in self._ref_counts:
                    del self._ref_counts[name]

                if name in self._alloc:
                    # return to free list
                    self._free.append(name)
                    return
            finally:
                self._lock.release()
        
        # If lock failed or not in pool, try ephemeral unlink
        try:
            SharedMemoryStore.unlink(name)
        except Exception:
            pass

    def close(self) -> None:
        # unlink all allocated blocks
        # Use a short timeout for close to avoid hanging the whole system
        if self._lock.acquire(timeout=1.0):
            try:
                for name in list(self._alloc):
                    try:
                        shm = shared_memory.SharedMemory(name=name)
                        shm.unlink()
                        shm.close()
                    except Exception:
                        pass
                    with _SHM_LOCK:
                        if name in _ALL_CREATED_SHM:
                            _ALL_CREATED_SHM.remove(name)
                self._alloc[:] = []
                self._free[:] = []
                self._ref_counts.clear()
            finally:
                self._lock.release()
        
        with _SHM_LOCK:
            if self in _SHM_POOLS:
                _SHM_POOLS.remove(self)

    def get_status(self) -> dict:
        """Return pool usage statistics."""
        with self._lock:
            return {
                'total': len(self._alloc),
                'free': len(self._free),
                'used': len(self._alloc) - len(self._free),
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
def acquire_for_group(queue: BasePriorityQueue, desired_types, group: Optional[str] = None, timeout: Optional[float] = None, poll_interval: float = 0.01, partition: Optional[str] = None) -> Message:
    """Acquire a message atomically for a consumer group.

    This helper uses the queue's native `get` with a filter to ensure
    atomicity and fair competition among workers.
    """
    desired_set = set(desired_types)
    
    def combined_filter(m: Message) -> bool:
        # Match desired types
        if m.msg_type in desired_set:
            return True
        # Match shutdown targeted to this group (or all)
        if m.msg_type == 'shutdown':
            if group is None:
                return True
            if isinstance(m.payload, dict):
                target = m.payload.get('target')
                return target in (None, group)
        return False

    return _get(queue, block=True, timeout=timeout, msg_filter=combined_filter, partition=partition)


def load_array_from_payload(payload: dict, copy: bool = True) -> Optional[np.ndarray]:
    """Load an ndarray from a message payload by recursively searching for SHM metas.

    This helper is now fully symmetric with pack_payload: it scans the entire
    payload and returns the FIRST valid ndarray it finds (either direct or in SHM).
    """
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
                return SharedMemoryStore.put(obj)
        
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
    except Exception:
        return obj


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
        name = meta.get('name') or meta.get('shm_name')
        if not name:
            return
        try:
            if pool is not None and meta.get('pool'):
                pool.free(name)
            else:
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


def run_worker_loop(queue: BasePriorityQueue, business_fn, desired_types, group: Optional[str] = None, pool: Optional['SharedMemoryPool'] = None, worker_name: Optional[str] = None, processing_delay: float = 0.0, partition: Optional[str] = None, result_partition: Optional[str] = None, result_msg_type: str = 'result', on_task_start: Optional[callable] = None, on_task_done: Optional[callable] = None, stop_event: Optional[multiprocessing.Event] = None):
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
        stop_event: Event to signal graceful shutdown.
    """
    if pool is None:
        pool = get_default_pool()

    name = worker_name or f"worker-{os.getpid()}"
    logger.info(f"{name} started")
    try:
        while not (stop_event and stop_event.is_set()):
            try:
                msg = acquire_for_group(queue, desired_types, group=group, timeout=1.0, partition=partition)
            except Empty:
                continue

            if msg.msg_type == 'shutdown':
                logger.info(f"{name} shutting down")
                break

            if on_task_start:
                try:
                    on_start_res = on_task_start(msg)
                except Exception:
                    pass

            payload = msg.payload
            arr = None
            try:
                # load array if present
                arr = load_array_from_payload(payload)

                # call business logic
                try:
                    result = business_fn(arr, payload, msg)
                except Exception:
                    logger.exception(f"{name} business_fn error")
                    result = None

                if processing_delay:
                    time.sleep(processing_delay)

                if result is not None:
                    # delegate packing to the message-queue layer
                    packed_payload = pack_payload(result, pool)
                    res_msg = Message(payload=packed_payload, priority=50, msg_type=result_msg_type)
                    try:
                        _put(queue, res_msg, partition=result_partition)
                    except Exception:
                        # on failure, attempt fallback: send original result inline
                        res_msg = Message(payload=result, priority=50, msg_type=result_msg_type)
                        _put(queue, res_msg, partition=result_partition)
            finally:
                # free input buffer even if business_fn or packing fails
                free_payload(payload, pool=pool)
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
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    demo(num_workers=4, num_tasks=8, cascade_levels=3)

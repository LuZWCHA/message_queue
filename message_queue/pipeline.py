"""
High-Level Pipeline Framework for Multi-Process Data Processing.

This module provides a decorator-based API for building complex data pipelines.
It handles process management, inter-node communication, shared memory 
optimization, and real-time monitoring.

Key Components:
- @node: Decorator to transform a function into a pipeline stage.
- Pipeline: Orchestrator that manages workers, queues, and metrics.
- PipelineNode: Base class for custom node implementations.
"""
from __future__ import annotations
import multiprocessing
import logging
import time
import os
import signal
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Union, Iterable, Callable

from .message_queue import (
    PartitionedPriorityQueue, Message, SharedMemoryPool, 
    run_worker_loop, set_default_pool, get_default_pool,
    pack_payload, free_payload, load_array_from_payload,
    _cleanup_shm, set_shm_registry, set_shm_master
)
from .monitor import start_monitor_server

logger = logging.getLogger(__name__)

class PipelineNode(ABC):
    """
    Abstract base class for a pipeline stage.
    
    A node defines how data is processed and how many workers should be 
    allocated to it. It also provides lifecycle hooks (setup/teardown).
    
    Attributes:
        name: Unique identifier for the node.
        num_workers: Number of parallel processes for this stage.
        input_partition: The queue partition to read from.
        output_partition: The queue partition to write results to.
        result_msg_type: The type tag for output messages.
        priority: Default priority for output messages.
        processing_delay: Artificial delay for simulation/testing.
    """
    
    def __init__(self, name: str, num_workers: int = 1, input_partition: Optional[str] = None, 
                 output_partition: Optional[str] = None, result_msg_type: str = 'task',
                 priority: int = 50, processing_delay: float = 0.0, capacity: Optional[int] = None, **kwargs):
        self.name = name
        self.num_workers = num_workers
        self.input_partition = input_partition
        self.output_partition = output_partition
        self.result_msg_type = result_msg_type
        self.priority = priority
        self.processing_delay = processing_delay
        self.capacity = capacity
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def log(self, level: int, msg: str, *args, **kwargs):
        """Log a message with node context."""
        extra = kwargs.get('extra', {})
        extra.update({'node': self.name, 'pid': os.getpid()})
        kwargs['extra'] = extra
        self.logger.log(level, f"[{self.name}] {msg}", *args, **kwargs)

    def info(self, msg: str, *args, **kwargs): self.log(logging.INFO, msg, *args, **kwargs)
    def error(self, msg: str, *args, **kwargs): self.log(logging.ERROR, msg, *args, **kwargs)
    def warning(self, msg: str, *args, **kwargs): self.log(logging.WARNING, msg, *args, **kwargs)
    def debug(self, msg: str, *args, **kwargs): self.log(logging.DEBUG, msg, *args, **kwargs)

    def setup(self, worker_id: int = 0):
        """Initialize resources (e.g., load models). Called once per worker process."""
        pass

    @abstractmethod
    def process(self, data: Any, payload: dict, msg: Message) -> Optional[Any]:
        """Process a single message.
        
        Args:
            data: The unwrapped data (e.g., numpy array from SHM if present).
            payload: The full message payload dictionary.
            msg: The original Message object.
            
        Returns:
            The result to be sent to the next stage, or None to drop.
        """
        pass

    def teardown(self):
        """Cleanup resources. Called once per worker process on shutdown."""
        pass

class SimpleNode(PipelineNode):
    """A simple pipeline node that wraps a function."""
    def __init__(self, fn: callable, name: Optional[str] = None, 
                 setup_fn: Optional[callable] = None, 
                 teardown_fn: Optional[callable] = None, **kwargs):
        name = name or fn.__name__
        self.context = kwargs.pop('context', None)
        super().__init__(name=name, **kwargs)
        self.fn = fn
        self.setup_fn = setup_fn
        self.teardown_fn = teardown_fn
        import inspect
        self._params_count = len(inspect.signature(fn).parameters)

    def setup(self, worker_id: int = 0):
        """Called by worker process."""
        if self.setup_fn:
            import inspect
            sig = inspect.signature(self.setup_fn)
            params = list(sig.parameters.keys())
            
            if len(params) > 0:
                args = []
                # First argument is worker_id
                args.append(worker_id)
                
                # For the rest, try to find them in node attributes
                for param_name in params[1:]:
                    if hasattr(self, param_name):
                        args.append(getattr(self, param_name))
                    else:
                        # If not found, we might have a problem, but let's try to call it anyway
                        # or maybe it's a default argument
                        pass
                
                try:
                    self.context = self.setup_fn(*args)
                except TypeError:
                    # Fallback to just worker_id if positional args fail
                    self.context = self.setup_fn(worker_id)
            else:
                self.context = self.setup_fn()

    def teardown(self, fn: Optional[callable] = None):
        """Can be used as a decorator or called by worker."""
        if fn is not None:
            self.teardown_fn = fn
            return fn
        if self.teardown_fn:
            if self.context is not None:
                self.teardown_fn(self.context)
            else:
                self.teardown_fn()

    def process(self, data: Any, payload: dict, msg: Message) -> Optional[Any]:
        args = [data]
        if self.context is not None:
            if self._params_count >= 2: args.append(self.context)
            if self._params_count >= 3: args.append(payload)
            if self._params_count >= 4: args.append(msg)
        else:
            if self._params_count >= 2: args.append(payload)
            if self._params_count >= 3: args.append(msg)
        
        return self.fn(*args[:self._params_count])

def node(name: Optional[str] = None, workers: int = 1, input: Optional[str] = None, 
         output: Optional[str] = None, delay: float = 0.0, capacity: Optional[int] = None,
         setup: Optional[callable] = None, teardown: Optional[callable] = None):
    """
    Decorator to create a pipeline node from a function.
    
    The decorated function can have various signatures:
    - fn(data)
    - fn(data, context)
    - fn(data, context, payload)
    - fn(data, context, payload, message)
    
    Usage:
        @node(name="preprocess", workers=2, output="inference", capacity=100)
        def preprocess(image):
            return image / 255.0

        @preprocess.setup
        def init():
            return "some_context"
            
    Args:
        name: Node name (defaults to function name).
        workers: Number of parallel processes.
        input: Input partition name.
        output: Output partition name.
        delay: Artificial delay (seconds).
        capacity: Queue capacity for the input partition.
        setup: Optional setup function.
        teardown: Optional teardown function.
    """
    def decorator(fn):
        node_obj = SimpleNode(
            fn=fn, 
            name=name or fn.__name__, 
            num_workers=workers, 
            input_partition=input, 
            output_partition=output,
            processing_delay=delay,
            capacity=capacity,
            setup_fn=setup,
            teardown_fn=teardown
        )
        # Attach the node object to the function so Pipeline.add can find it
        fn._pipeline_node = node_obj
        
        # Add setup/teardown helper decorators to the function as well
        def setup_decorator(setup_fn):
            node_obj.setup_fn = setup_fn
            return setup_fn
        fn.setup = setup_decorator
        
        def teardown_decorator(teardown_fn):
            node_obj.teardown_fn = teardown_fn
            return teardown_fn
        fn.teardown = teardown_decorator
        
        return fn
    return decorator

class Pipeline:
    """
    Orchestrator for a multi-stage processing pipeline.
    
    The Pipeline class manages the lifecycle of all nodes, handles the 
    Shared Memory Pool, and provides monitoring capabilities.
    
    Key Features:
    - Automatic Worker Management: Restarts dead workers automatically.
    - SHM Leak Prevention: Reclaims shared memory from crashed workers.
    - Real-time Dashboard: Built-in web server for performance metrics.
    - Fast Shutdown: Parallel process termination for near-instant stop.
    """
    
    def __init__(self, manager: Optional[Any] = None, 
                 pool_size: int = 16, block_size: int = 1024*1024*3,
                 enable_metrics: bool = True, metrics_sample_rate: float = 1.0, metrics_sample_every: int = 1):
        self._manager = manager or multiprocessing.Manager()
        self._nodes: List[PipelineNode] = []
        self._pool_size = pool_size
        self._block_size = block_size
        self._queue: Optional[PartitionedPriorityQueue] = None
        self._pool: Optional[SharedMemoryPool] = None
        self._processes: Dict[int, tuple] = {} # pid -> (node, process_obj)
        self._active_tasks = self._manager.dict() # pid -> Message
        self._stop_event = multiprocessing.Event()

        # Restart backoff (exponential) to avoid thrashing when workers keep dying
        self._restart_backoff = 0.0
        self._restart_backoff_base = 0.1
        self._restart_backoff_max = 5.0

        # Metrics sampling controls
        self.enable_metrics = enable_metrics
        self.metrics_sample_rate = metrics_sample_rate
        self.metrics_sample_every = metrics_sample_every
        
        # Metrics for monitoring
        self.metrics = self._manager.dict()
        self.metrics["nodes"] = {}
        self.metrics["pipeline"] = {
            "input_count": 0,
            "output_count": 0,
            "input_rate": 0.0,
            "output_rate": 0.0
        }
        self.metrics["pool"] = {"used": 0, "total": 0}
        self.metrics["partitions"] = {}
        
        self._last_metrics_snapshot = {}
        self._monitor_thread = None
        self._rate_thread = None

    @classmethod
    def from_nodes(cls, nodes: List[Union[PipelineNode, callable]], **kwargs) -> Pipeline:
        """Create a pipeline from a list of nodes/functions."""
        p = cls(**kwargs)
        for n in nodes:
            p.add(n)
        return p

    def start_monitor(self, port=8000):
        """Start the web monitor server and rate calculation thread."""
        if self._monitor_thread is not None:
            return
            
        self._monitor_thread = threading.Thread(
            target=start_monitor_server, 
            args=(self.metrics, port),
            daemon=True
        )
        self._monitor_thread.start()
        
        self._rate_thread = threading.Thread(target=self._rate_calculator_loop, daemon=True)
        self._rate_thread.start()
        logger.info(f"Monitor started at http://localhost:{port}")

    def _rate_calculator_loop(self):
        """Periodically calculate rates (TPS) for all nodes."""
        last_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                time.sleep(1.0)
                now = time.time()
                dt = now - last_time
                if dt <= 0:
                    continue
                
                # Update pipeline rates
                p_metrics = self.metrics.get("pipeline") if hasattr(self.metrics, "get") else None
                if p_metrics is None:
                    break
                in_count = p_metrics.get("input_count", 0)
                out_count = p_metrics.get("output_count", 0)
                
                last_in = self._last_metrics_snapshot.get("in", 0)
                last_out = self._last_metrics_snapshot.get("out", 0)
                
                p_metrics["input_rate"] = round((in_count - last_in) / dt, 2)
                p_metrics["output_rate"] = round((out_count - last_out) / dt, 2)
                
                self._last_metrics_snapshot["in"] = in_count
                self._last_metrics_snapshot["out"] = out_count
                self.metrics["pipeline"] = p_metrics # Trigger sync
                
                # Update node rates
                n_metrics = self.metrics.get("nodes") if hasattr(self.metrics, "get") else None
                if n_metrics is None:
                    break
                for node_name, stats in n_metrics.items():
                    curr_total = stats.get("success", 0) + stats.get("fail", 0)
                    last_total = self._last_metrics_snapshot.get(f"node_{node_name}", 0)
                    stats["rate"] = round((curr_total - last_total) / dt, 2)
                    self._last_metrics_snapshot[f"node_{node_name}"] = curr_total

                    curr_produced = stats.get("produced", 0)
                    last_produced = self._last_metrics_snapshot.get(f"node_{node_name}_prod", 0)
                    stats["produced_rate"] = round((curr_produced - last_produced) / dt, 2)
                    self._last_metrics_snapshot[f"node_{node_name}_prod"] = curr_produced

                    # Update worker liveness counts
                    alive = 0
                    for pid, (proc_node, proc, *_) in list(self._processes.items()):
                        if proc_node.name == node_name and proc.is_alive():
                            alive += 1
                    stats["workers_alive"] = alive
                    
                    n_metrics[node_name] = stats # Trigger sync
                
                self.metrics["nodes"] = n_metrics

                # Update Pool status
                if self._pool:
                    pool_status = self._pool.get_status()
                    self.metrics["pool"] = pool_status

                # Update Partition status
                if self._queue:
                    q_status = self._queue.get_status()
                    self.metrics["partitions"] = q_status

                last_time = now
            except Exception:
                if self._stop_event.is_set():
                    break
                logger.debug("Rate calculator loop terminating after exception", exc_info=True)
                break

    def wait_for_completion(self, timeout: Optional[float] = None, check_interval: float = 1.0):
        """Wait until all queues are empty and no workers are processing."""
        start_time = time.time()
        logger.info("Waiting for pipeline completion...")
        while not self._stop_event.is_set():
            # Check if any messages are still in the queue
            status = self._queue.get_status()
            total_queued = sum(p['size'] for p in status.values())
            
            # Check if any worker is still processing a task
            active_workers = len(self._active_tasks)
            
            if total_queued == 0 and active_workers == 0:
                # Double check after a short delay to avoid race conditions
                time.sleep(0.2)
                status = self._queue.get_status()
                total_queued = sum(p['size'] for p in status.values())
                active_workers = len(self._active_tasks)
                if total_queued == 0 and active_workers == 0:
                    logger.info("Pipeline completed all tasks.")
                    break
                
            if timeout and (time.time() - start_time > timeout):
                logger.warning("Timeout waiting for pipeline completion.")
                break
                
            time.sleep(check_interval)

    def add_node(self, node: Union[PipelineNode, callable], **kwargs) -> Pipeline:
        if not isinstance(node, PipelineNode) and callable(node):
            if hasattr(node, "_pipeline_node"):
                node = getattr(node, "_pipeline_node")
            else:
                node = SimpleNode(node)
        
        # Apply extra configuration
        for k, v in kwargs.items():
            setattr(node, k, v)
            
        self._nodes.append(node)
        
        # Initialize node metrics
        n_metrics = self.metrics["nodes"]
        n_metrics[node.name] = {
            "success": 0,
            "fail": 0,
            "produced": 0,
            "rate": 0.0,
            "produced_rate": 0.0,
            "latency": 0.0,
            "input": node.input_partition,
            "output": node.output_partition,
            "pct_get": 0.0,
            "pct_put": 0.0,
            "pct_fn": 0.0,
            "workers_total": node.num_workers,
            "workers_alive": 0
        }
        self.metrics["nodes"] = n_metrics
        
        logger.info(f"Added node '{node.name}' to pipeline (workers: {node.num_workers}, input: {node.input_partition})")
        return self

    def add(self, node: Union[PipelineNode, callable], **kwargs) -> Pipeline:
        """Alias for add_node."""
        return self.add_node(node, **kwargs)

    def run(self, monitor_port: Optional[int] = 8000, capacities: Optional[Dict[str, int]] = None, check_interval: float = 1.0):
        """
        High-level entry point: starts monitor, starts workers, and enters 
        the monitoring loop until interrupted.
        """
        self._stop_requested = 0
        
        def handle_exit(signum, frame):
            self._stop_requested += 1
            if self._stop_requested > 1:
                # Second Ctrl+C: Force exit immediately
                # Try to cleanup SHM before hard exit
                try:
                    from .message_queue import _cleanup_shm
                    _cleanup_shm()
                except Exception:
                    pass
                os._exit(1)
            self._stop_event.set()
        
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)

        try:
            if monitor_port:
                self.start_monitor(port=monitor_port)
            
            if not self._queue:
                self.build(capacities=capacities)
            
            self.start()
            
            logger.info("Pipeline is running. Press Ctrl+C to stop (twice to force exit).")
            while not self._stop_event.is_set():
                self.monitor_and_restart()
                time.sleep(check_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    @staticmethod
    def _worker_entry(node: PipelineNode, queue: PartitionedPriorityQueue, pool: SharedMemoryPool, 
                      active_tasks: dict, metrics: dict, stop_event: Any, shm_registry: Optional[List[str]] = None, worker_id: int = 0,
                      enable_metrics: bool = True, metrics_sample_rate: float = 1.0, metrics_sample_every: int = 1):
        """Entry point for worker processes."""
        set_default_pool(pool)
        if shm_registry is not None:
            set_shm_registry(shm_registry)
        set_shm_master(False) # Workers are never SHM masters

        # Defensive: if metrics proxy was corrupted or replaced, disable metrics updates
        metrics_proxy = metrics if hasattr(metrics, "get") else None
            
        pid = os.getpid()
        
        # Setup signal handlers for workers to ensure cleanup on SIGTERM
        def handle_sigterm(*args):
            try:
                node.teardown()
            except Exception:
                pass
            _cleanup_shm()
            os._exit(0)
        signal.signal(signal.SIGTERM, handle_sigterm)

        def on_start(msg):
            active_tasks[pid] = msg
            msg.payload['_start_time'] = time.time()
            
        def on_produce(item):
            try:
                if metrics_proxy is None:
                    return
                n_metrics = metrics_proxy.get("nodes")
                if n_metrics is not None:
                    stats = n_metrics.get(node.name)
                    if stats is not None:
                        stats["produced"] += 1
                        n_metrics[node.name] = stats
                        metrics_proxy["nodes"] = n_metrics
            except Exception:
                pass

        def on_done(msg):
            if pid in active_tasks:
                del active_tasks[pid]
            
            duration = time.time() - msg.payload.get('_start_time', time.time())
            t_get = t_put = t_fn = total_t = 0.0
            if isinstance(msg.payload, dict):
                item_cnt = msg.payload.get('_generator_item_count', 0)
                if item_cnt:
                    duration = duration / max(item_cnt, 1)
                t_get = msg.payload.get('_t_get', 0.0)
                t_put = msg.payload.get('_t_put', 0.0)
                t_fn = msg.payload.get('_t_fn', 0.0)
                total_t = t_get + t_put + t_fn
            
            # Update metrics
            try:
                if metrics_proxy is None:
                    return
                n_metrics = metrics_proxy.get("nodes")
                if n_metrics is not None:
                    stats = n_metrics.get(node.name)
                    if stats is not None:
                        # stats is a regular dict (copy from Manager.dict)
                        if msg.payload.get('error'):
                            stats["fail"] += 1
                        else:
                            stats["success"] += 1
                        
                        # Moving average for latency
                        curr_lat = stats.get("latency", 0.0)
                        stats["latency"] = round(curr_lat * 0.9 + duration * 0.1, 4)

                        # Time proportion smoothing
                        if isinstance(msg.payload, dict):
                            if total_t > 0:
                                pct_get = t_get / total_t
                                pct_put = t_put / total_t
                                pct_fn = t_fn / total_t
                                stats["pct_get"] = round(stats.get("pct_get", 0.0) * 0.9 + pct_get * 0.1, 4)
                                stats["pct_put"] = round(stats.get("pct_put", 0.0) * 0.9 + pct_put * 0.1, 4)
                                stats["pct_fn"] = round(stats.get("pct_fn", 0.0) * 0.9 + pct_fn * 0.1, 4)
                        
                        n_metrics[node.name] = stats # Update the Manager.dict
                        metrics_proxy["nodes"] = n_metrics # Ensure top-level sync

                        # If this node is a sink (no output partition), count as pipeline output
                        if not node.output_partition:
                            p_metrics = metrics_proxy.get("pipeline")
                            if p_metrics is not None:
                                p_metrics["output_count"] += 1
                                metrics_proxy["pipeline"] = p_metrics
            except Exception as e:
                logger.error(f"Metrics update error in worker: {e}")

        try:
            # Define a filter that only accepts messages targeted to this worker_id or untargeted messages
            def worker_msg_filter(m: Message) -> bool:
                target_id = m.meta.get('target_worker_id')
                return target_id is None or target_id == worker_id

            node.setup(worker_id=worker_id)
            run_worker_loop(
                queue=queue,
                business_fn=node.process,
                desired_types=(node.name + "_task", "task"),
                group=node.name,
                pool=pool,
                worker_name=f"{node.name}-{pid}",
                processing_delay=node.processing_delay,
                partition=node.input_partition,
                result_partition=node.output_partition,
                result_msg_type=node.result_msg_type,
                on_task_start=on_start,
                on_task_done=on_done,
                on_produce=on_produce,
                stop_event=stop_event,
                enable_metrics=enable_metrics,
                metrics_sample_rate=metrics_sample_rate,
                metrics_sample_every=metrics_sample_every,
                msg_filter=worker_msg_filter
            )
        except Exception as e:
            node.error(f"Crashed with exception: {e}", exc_info=True)
            raise
        finally:
            node.teardown()

    def build(self, capacities: Optional[Dict[str, int]] = None):
        """Initialize queue and pool based on added nodes."""
        # Determine partitions and capacities
        parts = capacities or {}
        
        # First pass: collect all input partition capacities (highest priority)
        for node in self._nodes:
            if node.input_partition:
                cap = node.capacity or max(node.num_workers * 2, 16)
                # Input partition capacity always wins
                parts[node.input_partition] = cap
        
        # Second pass: set default capacities for output partitions that aren't inputs
        for node in self._nodes:
            if node.output_partition and node.output_partition not in parts:
                parts[node.output_partition] = 1024 # Default large capacity for results/meta

        self._queue = PartitionedPriorityQueue(manager=self._manager, partitions=parts, stop_event=self._stop_event)
        
        # Initialize shared SHM registry
        self._shm_registry = self._manager.list()
        set_shm_registry(self._shm_registry)
        set_shm_master(True)
        
        self._pool = SharedMemoryPool.create(self._manager, pool_size=self._pool_size, block_size=self._block_size, stop_event=self._stop_event)
        set_default_pool(self._pool)
        return self

    def _start_worker(self, node: PipelineNode, worker_id: int = 0):
        p = multiprocessing.Process(
            target=self._worker_entry, 
            args=(node, self._queue, self._pool, self._active_tasks, self.metrics, self._stop_event, 
                  getattr(self, '_shm_registry', None), worker_id, 
                  self.enable_metrics, self.metrics_sample_rate, self.metrics_sample_every),
            daemon=True
        )
        p.start()
        self._processes[p.pid] = (node, p, worker_id)
        return p

    def start(self):
        """Launch all worker processes."""
        if not self._queue:
            self.build()
            
        for node in self._nodes:
            for i in range(node.num_workers):
                self._start_worker(node, worker_id=i)
        logger.info(f"Pipeline started with {len(self._nodes)} nodes and {len(self._processes)} workers.")

    def monitor_and_restart(self):
        """Check for dead workers, reclaim their SHM, and restart them."""
        if self._stop_event.is_set():
            return
            
        dead_pids = []
        for pid, info in list(self._processes.items()):
            node, p = info[0], info[1]
            if not p.is_alive():
                logger.warning(f"Worker {node.name} (pid {pid}) died! Reclaiming SHM and restarting...")
                dead_pids.append(pid)
        
        for pid in dead_pids:
            # 1. Reclaim SHM if the worker was processing a task
            info = self._processes.pop(pid)
            node, _, worker_id = info

            if pid in self._active_tasks:
                msg = self._active_tasks.pop(pid)
                try:
                    # Tag the message so only the restarted worker with the same worker_id can pick it up
                    if msg.meta is None:
                        msg.meta = {}
                    msg.meta['target_worker_id'] = worker_id
                    
                    # Put the in-flight task back to the same partition
                    self._queue.put(msg, partition=node.input_partition, block=False)
                    logger.info(f"Requeued in-flight task {msg.id} from dead worker {pid} (worker_id {worker_id}) into partition {node.input_partition}")
                except Exception as e:
                    logger.error(f"Failed to requeue task {msg.id} from dead worker {pid}: {e}")
                    try:
                        free_payload(msg.payload, pool=self._pool)
                    except Exception:
                        pass
            else:
                logger.info(f"No active task found for dead worker {pid} (worker_id {worker_id})")

            # 2. Restart worker
            # Exponential backoff to avoid restart thrash; reset when stable
            if self._restart_backoff > 0:
                time.sleep(self._restart_backoff)
            self._start_worker(node, worker_id=worker_id)
            next_backoff = self._restart_backoff * 2 if self._restart_backoff > 0 else self._restart_backoff_base
            self._restart_backoff = min(self._restart_backoff_max, next_backoff)

        if not dead_pids:
            # Reset backoff when the system is stable
            self._restart_backoff = 0.0

    def get_pool_status(self) -> dict:
        if self._pool:
            return self._pool.get_status()
        return {}

    def put(self, payload: Any, partition: str, msg_type: str = 'task', priority: int = 50):
        """Inject a message into the pipeline."""
        if not self._queue:
            self.build()
            
        msg = Message(payload=payload, priority=priority, msg_type=msg_type)
        # If payload contains numpy arrays, pack them
        packed = pack_payload(msg.payload, self._pool)
        msg.payload = packed
        self._queue.put(msg, partition=partition)
        
        # Update metrics
        p_metrics = self.metrics.get("pipeline")
        if p_metrics is not None:
            p_metrics["input_count"] += 1
            self.metrics["pipeline"] = p_metrics

    def get_result(self, partition: str, timeout: Optional[float] = None, msg_filter: Optional[callable] = None) -> Optional[Message]:
        """Fetch a result from the pipeline."""
        if not self._queue:
            self.build()
            
        try:
            msg = self._queue.get(partition=partition, block=True, timeout=timeout, msg_filter=msg_filter)
        except Exception: # Catch Empty or other queue errors
            return None
            
        if msg:
            # Update metrics
            p_metrics = self.metrics.get("pipeline")
            if p_metrics is not None:
                p_metrics["output_count"] += 1
                self.metrics["pipeline"] = p_metrics
            
            # Automatically free SHM for the result message
            free_payload(msg.payload, pool=self._pool)
            
        return msg

    def stop(self):
        """
        Gracefully shut down all nodes.
        
        The shutdown process follows these steps:
        1. Set the global stop event to signal all loops to exit.
        2. Send high-priority 'shutdown' messages to all partitions.
        3. Wait in parallel for all processes to exit (up to 2 seconds).
        4. Forcefully terminate any remaining processes.
        5. Cleanup Shared Memory and Queues.
        """
        if getattr(self, '_stopping', False):
            return
        self._stopping = True
        
        logger.info("Stopping pipeline...")
        self._stop_event.set()
        
        # 1. Send shutdown messages to all partitions (non-blocking)
        if self._queue:
            for node in self._nodes:
                for _ in range(node.num_workers):
                    try:
                        shutdown_msg = Message(payload={'target': node.name}, priority=1000, msg_type='shutdown')
                        self._queue.put(shutdown_msg, partition=node.input_partition, block=False)
                    except Exception:
                        pass
        
        # 2. Wait for processes to exit gracefully (parallel wait)
        start_wait = time.time()
        grace_period = 1.0 # Reduced grace period
        while time.time() - start_wait < grace_period:
            alive_pids = [pid for pid, (node, p, *_) in self._processes.items() if p.is_alive()]
            if not alive_pids:
                break
            time.sleep(0.1)
            
        # 3. Terminate remaining processes
        for pid, (node, p, *_) in self._processes.items():
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        
        # 4. Final join with short timeout and kill fallback
        for pid, (node, p, *_) in self._processes.items():
            try:
                p.join(timeout=0.1)
                if p.is_alive():
                    p.kill() # SIGKILL
                    p.join(timeout=0.1)
            except Exception:
                pass
        
        # 5. Cleanup resources (with error handling to avoid hangs)
        try:
            if self._queue:
                # Free all payloads still in the queue before closing the pool
                self._queue.cleanup_payloads(pool=self._pool)
        except Exception as e:
            logger.warning(f"Error cleaning up queue payloads: {e}")

        try:
            if self._pool:
                # pool.close() now has internal timeouts
                self._pool.close()
        except Exception as e:
            logger.warning(f"Error closing pool: {e}")

        try:
            if self._queue:
                self._queue.close()
        except Exception as e:
            logger.warning(f"Error closing queue: {e}")
            
        # 6. Final global cleanup for this process and all registered SHM
        if hasattr(self, '_shm_registry') and self._shm_registry is not None:
            try:
                # Use a local copy to avoid hanging on Manager if it's already shutting down
                # We use a short timeout-like approach by checking if manager is alive
                # but since we can't easily check manager health, we just wrap in try-except
                shm_names = list(self._shm_registry)
                if shm_names:
                    logger.info(f"Cleaning up {len(shm_names)} registered SHM segments...")
                    for name in shm_names:
                        try:
                            from multiprocessing import shared_memory
                            shm = shared_memory.SharedMemory(name=name)
                            shm.unlink()
                            shm.close()
                        except Exception:
                            pass
                self._shm_registry[:] = []
            except Exception:
                # If manager is dead, we can't access the registry, just skip
                pass

        _cleanup_shm()
            
        logger.info("Pipeline stopped.")

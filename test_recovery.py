import multiprocessing
import time
import os
import signal
import logging
from message_queue.pipeline import Pipeline, node

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(process)d] %(message)s')
logger = logging.getLogger(__name__)

@node(name="worker_node", workers=2, input="input_stage", output="output_stage")
def processing_node(data, context, payload, msg):
    worker_id = context
    task_data = payload
    
    # Use a shared counter file to limit total crashes
    crash_counter_file = "/tmp/crash_total_count"
    
    target_worker = msg.meta.get('target_worker_id')
    if target_worker is not None:
        logger.info(f"Worker {worker_id} RECEIVED REQUEUED TASK {msg.id} (was targeted to {target_worker})")
    else:
        logger.info(f"Worker {worker_id} processing task {task_data}")

    # Logic to crash multiple times (e.g., 3 times total)
    should_crash = False
    try:
        if not os.path.exists(crash_counter_file):
            count = 0
        else:
            with open(crash_counter_file, "r") as f:
                count = int(f.read().strip())
        
        if count < 10: # Crash 10 times total
            should_crash = True
            with open(crash_counter_file, "w") as f:
                f.write(str(count + 1))
    except Exception:
        pass

    if should_crash:
        logger.warning(f"Worker {worker_id} is about to CRASH intentionally (Crash #{count+1}) while processing {task_data}...")
        # Random delay to make crashes unpredictable
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGKILL)
    
    time.sleep(0.5)
    return f"Result of {task_data}"

@processing_node.setup
def setup_node(worker_id):
    logger.info(f"Worker {worker_id} setting up")
    return worker_id

def test_recovery():
    # Clean up crash trigger
    if os.path.exists("/tmp/crash_total_count"):
        os.remove("/tmp/crash_total_count")

    pipeline = Pipeline(pool_size=10)
    pipeline.add(processing_node)
    pipeline.start()
    
    num_tasks = 6
    for i in range(num_tasks):
        pipeline.put(f"Task-{i+1}", partition="input_stage")

    logger.info(f"{num_tasks} tasks injected. Monitoring for multiple crashes and recovery...")
    
    start_time = time.time()
    results = []
    
    try:
        # Wait longer for multiple restarts
        while time.time() - start_time < 30:
            pipeline.monitor_and_restart()
            
            res_msg = pipeline.get_result("output_stage", timeout=0.5)
            if res_msg:
                logger.info(f"Main received: {res_msg.payload}")
                results.append(res_msg.payload)
            
            if len(results) >= num_tasks:
                logger.info("All tasks completed successfully after multiple crashes!")
                break
                
            time.sleep(0.5)
            
    finally:
        pipeline.stop()
        if os.path.exists("/tmp/crash_total_count"):
            os.remove("/tmp/crash_total_count")

    if len(results) == num_tasks:
        print(f"\nSUCCESS: System recovered from multiple crashes. Processed {len(results)} tasks.")
    else:
        print(f"\nFAILED: Only processed {len(results)}/{num_tasks} tasks.")

if __name__ == "__main__":
    test_recovery()

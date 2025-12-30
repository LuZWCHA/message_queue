import time
import os
import numpy as np
from PIL import Image
import logging
import multiprocessing
from message_queue.pipeline import Pipeline, node

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger("ModernDemo")

# --- Node Definitions ---

@node(workers=2, input="crop_queue", output="infer_queue")
def crop_node(data, payload):
    """
    模拟切图节点：读取大图并切块。
    返回的字典中包含 'data' (numpy array)，Pipeline 会自动将其放入共享内存。
    """
    # print(f"PROCESS: crop_node slide {payload.get('slide_id')}", flush=True)
    # 模拟读取一张 2048x2048 的大图
    slide_size = (2048, 2048)
    # 只有在第一次运行或模拟时生成，实际场景中这里可能是 OpenSlide 读取
    arr = (np.random.rand(slide_size[1], slide_size[0], 3) * 255).astype('uint8')
    slide = Image.fromarray(arr)
    
    coord = payload.get('coord')

    for x,y,w,h in [coord] * 20:  # 模拟每个任务切 10 个块
        x, y, w, h = coord
        crop = slide.crop((x, y, x + w, y + h))
        crop_arr = np.asarray(crop)
        time.sleep(0.05)  # 模拟切图耗时
    
        yield {
            'slide_id': payload.get('slide_id'),
            'coord': coord,
            'data': crop_arr  # 自动进入 SHM
        }

@node(workers=2, input="infer_queue", output="results")
def infer_node(data, payload):
    """
    模拟推理节点：接收切块数据并进行“模型推理”。
    'data' 参数会自动从共享内存加载为 numpy array。
    """
    if data is None:
        return None
    
    # 模拟模型推理耗时
    time.sleep(0.05)
    
    # 简单的计算模拟推理结果
    score = float(np.mean(data)) / 255.0
    return {
        'slide_id': payload.get('slide_id'),
        'coord': payload.get('coord'),
        'score': score,
        'model': 'MockNet-v1'
    }

@node(workers=1, input="results")
def sink_node(data, payload):
    """
    落地节点：消费最终结果，防止队列堆积。
    """
    pass

# --- Execution ---

if __name__ == "__main__":
    # 强制使用 spawn 模式，避免 fork 带来的多线程锁问题
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    p = Pipeline.from_nodes(
        [crop_node, infer_node, sink_node],
        pool_size=2, 
        block_size=1024 * 1024  # 1MB blocks
    )
    
    import threading
    def producer():
        logger.info("Producer thread started.")
        
        # 模拟标准业务流
        num_slides = 20
        tile_size = 512
        for sidx in range(num_slides):
            if p._stop_event.is_set():
                break
            logger.info(f"Dispatching slide {sidx}...")
            for y in range(0, 2048, tile_size):
                for x in range(0, 2048, tile_size):
                    if p._stop_event.is_set():
                        break
                    p.put(
                        payload={'slide_id': sidx, 'coord': (x, y, tile_size, tile_size)}, 
                        partition="crop_queue"
                    )
                if p._stop_event.is_set():
                    break
            # time.sleep(1.0) # 模拟每秒处理一个 slide
        logger.info("Producer thread exiting.")
        p.put(None, partition="crop_queue", msg_type='shutdown')  # 发送结束信号
        
        logger.info("All test tasks dispatched.")

    threading.Thread(target=producer, daemon=True).start()
    
    # 3. 运行 Pipeline (包含 Web 监控、自动重启、指标计算)
    # 默认会进入监控循环，直到收到 Ctrl+C
    try:
        p.run(monitor_port=8001)
    except KeyboardInterrupt:
        logger.info("Main process interrupted.")

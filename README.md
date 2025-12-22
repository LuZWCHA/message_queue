# Message Queue Pipeline

A high-performance IPC and Shared Memory Pipeline infrastructure for Python.

## Features

- **Multi-priority Partitioned Queues**: Efficient inter-process communication with priority support.
- **Zero-copy Shared Memory**: Reference-counted shared memory pool for NumPy arrays.
- **Decorator-based API**: Easily transform functions into pipeline nodes using `@node`.
- **Real-time Monitoring**: Built-in web dashboard to visualize pipeline performance and topology.

## Installation

```bash
pip install .
```

## Requirements

- Python 3.8+
- NumPy

## Quick Start

```python
from message_queue import Pipeline, node

@node(name="producer", outputs=["data"])
def producer():
    for i in range(10):
        yield {"data": i}

@node(name="consumer", inputs=["data"])
def consumer(data):
    print(f"Received: {data}")

p = Pipeline()
p.add_node(producer)
p.add_node(consumer)
p.run()
```

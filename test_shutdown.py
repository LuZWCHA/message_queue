import subprocess
import time
import os
import signal

def test_shutdown():
    print("Starting pipeline demo...")
    # Use a smaller pool to trigger blocking more easily if needed
    process = subprocess.Popen(
        ["python3", "/nasdata2/private/zwlu/detection/CellDetection/message_queue/demo.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(5)
    print(f"Sending SIGINT to process {process.pid}...")
    process.send_signal(signal.SIGINT)
    
    try:
        stdout, stderr = process.communicate(timeout=10)
        print("Pipeline exited successfully.")
        print("STDOUT tail:\n", "\n".join(stdout.splitlines()[-10:]))
        print("STDERR tail:\n", "\n".join(stderr.splitlines()[-10:]))
    except subprocess.TimeoutExpired:
        print("CRITICAL: Pipeline HANGED after SIGINT!")
        process.kill()
        stdout, stderr = process.communicate()
        print("STDOUT tail:", "\n".join(stdout.splitlines()[-5:]))
        exit(1)

if __name__ == "__main__":
    test_shutdown()

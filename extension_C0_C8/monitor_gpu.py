"""Sample GPU usage while the authorized sequential experiment process exists."""
import csv,datetime,subprocess,time
from pathlib import Path
import psutil
here=Path(__file__).resolve().parent
# Shell sessions may isolate process inventories. Observe completion artifacts instead.
deadline=time.monotonic()+3600
path=here/'run_v2'/'gpu_monitor.csv'
with path.open('x',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['time_local','gpu_util_percent','used_memory_MiB','temperature_C','system_available_RAM_bytes'])
    while not (here/'run_v2'/'xai.json').exists() and time.monotonic()<deadline:
        v=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
        w.writerow([datetime.datetime.now().isoformat(),*[x.strip() for x in v],psutil.virtual_memory().available]);f.flush();time.sleep(10)
print('GPU monitoring completed',path)

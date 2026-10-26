"""Build fixed-Manager Answer data through the shared BF16/vLLM backend."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from verl_memory.train import main
if __name__ == '__main__': main(['--task','prepare',*sys.argv[1:]])

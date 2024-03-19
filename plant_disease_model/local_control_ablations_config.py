from control_ablations import config
from pathlib import Path

config.logging_root_path = Path(__file__).resolve().parent.parent
config.hsl_path = "/Users/racheltrimble/hsl/install/lib/libhsl.dylib"
import sys
import traceback
from importlib.machinery import SourceFileLoader

def load_module(file):
    try:
        return SourceFileLoader("x", file).load_module()
    except Exception:
        traceback.print_exc()
        return None

def print_traceback(tb):
    if tb is not None:
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No files provided")
        sys.exit(1)

    files = sys.argv[1:]
    had_failure = False
    for file in files:
        module = load_module(file)
        if module is None:
            had_failure = True
            print_traceback(module)
            print(f"Failed to load module from file: {file}")

    if not had_failure:
        print("All modules loaded successfully")
    sys.exit(1 if had_failure else 0)

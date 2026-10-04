import sys

from app.doctor import main as doctor


if __name__ == "__main__":
    if sys.argv[1:] == ["doctor"]:
        raise SystemExit(doctor())
    print("Usage: python -m app doctor")
    raise SystemExit(2)

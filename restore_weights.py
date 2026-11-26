"""合并仓库中的权重分片，并校验完整 SAM 3 权重。"""
import hashlib
import os
import tempfile
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent / "weights"
EXPECTED_HASH = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"


def main():
    """流式还原权重，通过校验后原子替换目标文件。"""
    destination = DIRECTORY / "sam3.pt"
    if destination.is_file():
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() == EXPECTED_HASH:
            print(f"权重校验通过 {destination}")
            return
        raise SystemExit("已有 sam3.pt 校验失败，请移走该文件后重试")
    parts = [DIRECTORY / f"sam3.pt.part{index:02d}" for index in range(1, 14)]
    if not all(path.is_file() for path in parts):
        raise SystemExit("缺少权重分片，请先执行 git lfs pull")
    digest = hashlib.sha256()
    descriptor, filename = tempfile.mkstemp(prefix="sam3-", suffix=".tmp", dir=DIRECTORY)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for part in parts:
                with part.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                        output.write(block)
        if digest.hexdigest() != EXPECTED_HASH:
            raise ValueError("分片校验失败，请确认已执行 git lfs pull")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"权重还原完成 {destination}")


if __name__ == "__main__":
    main()

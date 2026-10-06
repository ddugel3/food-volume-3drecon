"""GT 접근을 물리적으로 차단한 채 파이프라인 단계를 실행한다.

"GT 보면서 짜맞춘 것 아니냐"에 대한 답은 주장이 아니라 검증이어야 한다.

  python scripts/run_blind.py src/apply_shape.py --backend depthpro ...

구현 주의 -- 첫 판은 뚫렸다.
  이벤트 이름을 화이트리스트로 골라 감시했더니 Path.glob 이 통과했다.
  glob 은 open/os.listdir 이 아니라 **os.scandir** 을 낸다. 실측으로 GT 파일명이
  그대로 출력됐다. 이벤트 종류를 열거하는 방식은 빠뜨리는 게 생긴다.

  그래서 지금은 **모든 감사 이벤트의 모든 인자를 훑는다.** 문자열이든 PathLike 든
  시퀀스 안에 들어 있든, 금지 토큰이 보이면 그 자리에서 예외를 던진다.
  느리지만 확실하다. 빠뜨리는 쪽보다 낫다.
"""
import os
import runpy
import sys
from pathlib import Path

BLOCK = ("_held_out", "gt_ml", "gt_volumes", "ground_truth")
SKIP = ("sys.", "exec", "compile", "import", "object.", "builtins.",
        "code.", "marshal.", "pickle.", "gc.")
hits = []


def _scan(v, depth=0):
    if depth > 3:
        return None
    if isinstance(v, (str, bytes)):
        s = (v.decode("utf-8", "ignore") if isinstance(v, bytes) else v).lower()
        return s if any(b in s for b in BLOCK) else None
    if isinstance(v, os.PathLike):
        return _scan(os.fspath(v), depth + 1)
    if isinstance(v, (list, tuple, set)):
        for e in v:
            r = _scan(e, depth + 1)
            if r:
                return r
    if isinstance(v, dict):
        for e in v.values():
            r = _scan(e, depth + 1)
            if r:
                return r
    return None


def _hook(event, args):
    if event.startswith(SKIP):
        return
    for a in args:
        r = _scan(a)
        if r:
            hits.append((event, r))
            raise PermissionError(f"GT 접근 차단됨: {event} -> {r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("사용법: run_blind.py <스크립트> [인자...]")
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    sys.addaudithook(_hook)
    print(f"[blind] GT 차단 활성 · 금지 토큰 {BLOCK}", flush=True)
    try:
        runpy.run_path(target, run_name="__main__")
    finally:
        print(f"[blind] 종료 · GT 접근 시도 {len(hits)} 건", flush=True)

"""프로젝트 경로 상수.

파이프라인 코드는 data/work/ 만 본다. data/raw/ 를 직접 glob 하지 않는다 --
data/raw 아래에는 대회가 배포한 GT 메시가 들어 있었고(지금은 data/_HELD_OUT 로
격리했지만) 실수로 다시 들어올 수 있기 때문이다. HELD_OUT 은 최종 제출 확정 후
사후 오차분석에서만 연다. 자세한 것은 data/_HELD_OUT/README.md.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"
IMAGES = WORK / "images"
ID_MAP = ROOT / "data" / "id_map.csv"
CACHE = ROOT / "cache"
OUTPUTS = ROOT / "outputs"
SUBMISSIONS = ROOT / "submissions"

# 봉인 구역. 파이프라인에서 참조 금지 -- G1(사후 분석) 전용.
_HELD_OUT = ROOT / "data" / "_HELD_OUT"


def combo_images():
    """{combo_index: Path} -- 14개 장면 이미지."""
    return {int(p.stem.split("_")[1]): p for p in sorted(IMAGES.glob("combo_*"))}

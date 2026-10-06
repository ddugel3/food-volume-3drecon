"""필요한 모델 저장소에 실제로 접근할 수 있는지 확인한다.

model_info() 는 게이트 저장소에서도 성공하므로 접근 판정에 쓰면 안 된다.
반드시 실제 파일을 하나 받아봐야 한다.
"""
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

REPOS = [
    ("facebook/sam3", "SAM 3 — 텍스트 개념 세그멘테이션 (1순위)"),
    ("facebook/sam-3d-objects", "SAM 3D Objects — 단일이미지 3D (1순위)"),
    ("IDEA-Research/grounding-dino-base", "Grounding DINO — 텍스트->박스 (대체)"),
    ("facebook/sam2.1-hiera-large", "SAM 2.1 — 박스->마스크 (대체)"),
    ("Ruicheng/moge-3-vitl", "MoGe-3 — 메트릭 포인트맵"),
    ("Ruicheng/moge-2-vitl-normal", "MoGe-2 — 메트릭 포인트맵"),
    ("apple/DepthPro-hf", "Depth Pro — 메트릭 깊이 + 초점거리"),
    ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "Depth Anything V2 metric"),
    ("tencent/Hunyuan3D-2.1", "Hunyuan3D 2.1 — 이미지->3D"),
    ("microsoft/TRELLIS-image-large", "TRELLIS — 이미지->3D"),
]


def probe(rid):
    """HEAD 한 번으로 접근 가능 여부를 판정한다.

    hf_hub_download 로 '작은 json' 을 받아보는 방식은 오탐이 난다 -- MoGe 처럼
    저장소에 .pt 밖에 없으면 받을 파일이 없어서 '막힘' 으로 잘못 나온다.
    get_hf_file_metadata 는 실제 다운로드 없이 권한만 확인하므로 정확하고 빠르다.
    """
    api = HfApi()
    try:
        files = api.list_repo_files(rid)
    except Exception as e:
        return False, f"목록 실패 {type(e).__name__}"
    # README/LICENSE 는 게이트 저장소에서도 공개되므로 판정에 쓰면 안 된다.
    cand = [f for f in files
            if not f.startswith(".") and Path(f).name not in
            ("README.md", "LICENSE", "NOTICE", "Notice.txt", "CODE_OF_CONDUCT.md",
             "CONTRIBUTING.md")]
    if not cand:
        return False, "판정에 쓸 파일 없음"
    err = "?"
    for f in cand[:3]:
        try:
            get_hf_file_metadata(hf_hub_url(rid, f))
            return True, f
        except Exception as e:
            err = type(e).__name__
    return False, err


if __name__ == "__main__":
    try:
        from huggingface_hub import whoami
        print(f"로그인: {whoami()['name']}\n")
    except Exception:
        print("로그인: 익명 (게이트 저장소는 받을 수 없습니다)\n")
    for rid, desc in REPOS:
        ok, info = probe(rid)
        mark = "✔ 열림" if ok else "✘ 막힘"
        print(f"  {mark}  {rid:56s} {desc}")
        if not ok:
            print(f"          -> {info}. https://huggingface.co/{rid} 에서 라이선스 동의 후"
                  f" bash scripts/install-hf-token.sh")

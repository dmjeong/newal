# newal

**Qwen 기반 로컬 멀티모달 코딩 어시스턴트** — 텍스트 · 이미지 · 동영상 입력을 받아 내 PC에서만 동작합니다.

Python · Visual Studio 2026 · 완전 오프라인 · 사용량 제한 없음 · 무료.

---

## 먼저 솔직하게: Codex / Claude Code보다 강해질 수 있나?

**순수 모델 지능으로는 못 이깁니다.** 로컬에서 돌릴 수 있는 35B-A3B 급 모델이 프런티어 모델의
추론 능력을 앞서는 일은 없습니다. 그렇게 주장하는 프로젝트는 믿지 마세요.

**하지만 스캐폴드로는 이길 수 있는 축이 분명히 있습니다.** newal이 실제로 노리는 지점:

| 축 | newal | Codex / Claude Code |
|---|---|---|
| **동영상 입력** | 네이티브 지원. 화면 녹화를 던지면 장면 전환 프레임을 뽑아 읽음 | 미지원 |
| **이미지 입력** | 네이티브 지원 | 지원 |
| **영구 프로젝트 기억** | SQLite에 세션 간 유지. 다음 실행 때 자동 주입 | 세션 종료 시 소멸 |
| **검증 루프** | 편집 후 테스트 자동 실행 → 실패 출력 보고 자가 수정 (최대 N회) | 부분적 |
| **비용 / 제한** | 0원, 무제한, 오프라인 | 종량 과금, 레이트 리밋 |
| **코드 유출** | 없음. 네트워크를 안 씀 | 외부 API 전송 |
| **원시 추론 능력** | 약함 | **강함** |

즉 전략은 *"더 똑똑한 모델"* 이 아니라 **"덜 똑똑한 모델을 더 좋은 루프에 넣기"** 입니다.
계획 → 실행 → 테스트 → 실패 분석 → 수정 루프는 한 방 정답률이 낮은 모델의 최종 결과물을
크게 끌어올립니다. 실제로 품질을 만드는 건 이 부분입니다.

---

## 모델 선택 (2026년 8월 기준)

Qwen3.5부터는 **별도의 `-VL` 계열이 없습니다.** 본체 모델이 이미 텍스트+이미지+동영상을
네이티브로 처리합니다 (early-fusion 멀티모달). `Qwen3-VL`은 구세대입니다.

| 모델 | 총/활성 파라미터 | 4bit VRAM | 권장 GPU | 비고 |
|---|---|---|---|---|
| **`Qwen/Qwen3.6-35B-A3B`** | 35B / 3B MoE | **~20GB** | RTX 4090·5090 (24–32GB) | **기본값. 에이전트 코딩 특화, 활성 3B라 빠름** |
| `Qwen/Qwen3.5-27B` | 27B dense | ~16GB | RTX 4090 | 밀집 모델, 추론 안정적이나 느림 |
| `Qwen/Qwen3.5-9B` | 9B dense | ~6GB | RTX 4070·4080 | 중급 PC |
| `Qwen/Qwen3.5-4B` | 4B dense | ~3GB | RTX 3060 12GB | 노트북 |
| `Qwen/Qwen3.5-2B` | 2B dense | ~2GB | 내장 GPU도 가능 | 최소 사양 |

전부 256K 컨텍스트 (YaRN으로 1M까지 확장 가능), thinking 모드 내장.

> **왜 35B-A3B가 기본값인가:** MoE라 토큰당 3B만 활성화됩니다. 27B dense와 VRAM은 비슷한데
> 생성 속도는 몇 배 빠릅니다. 에이전트 루프는 한 턴에 호출을 수십 번 하므로 속도가 곧 체감 품질입니다.

---

## 설치 (Windows + Visual Studio 2026)

### 1. 프로젝트 열기

```
File → Open → Project/Solution → newal.sln
```
폴더 모드(`File → Open → Folder`)로 열어도 `launch.vs.json`의 실행 구성이 그대로 잡힙니다.

### 2. 가상환경 + 의존성

VS의 **Python Environments** 창에서 `venv` 추가 후, 또는 터미널에서:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 추론 엔진 설치

```powershell
# NVIDIA GPU (권장)
pip install vllm

# 또는 SGLang
pip install "sglang[all]"
```

> **Windows에서 vLLM이 안 깔릴 때:** WSL2를 쓰거나, `configs/local.yaml`에
> `backend: {kind: transformers}`로 설정하세요. 느리지만 서버 없이 동작합니다.
> (`pip install "newal[transformers]"` 필요)

### 4. 실행

VS 상단 실행 버튼 드롭다운에서 선택:
- **newal: chat (interactive)** — 어시스턴트 시작
- **newal: start Qwen server only** — 추론 서버만 띄우기
- **newal: run tests** — 테스트 실행

터미널에서는:

```powershell
python -m newal chat
```

첫 실행 시 모델 가중치를 자동으로 내려받습니다 (35B-A3B 기준 수십 GB, 시간이 걸립니다).
vLLM 서버도 자동으로 뜹니다.

---

## 사용법

```
> UserService의 인증 로직을 리팩터링해줘

> /img C:\Users\me\Desktop\error.png
> 이 에러 화면 원인이 뭐야?

> /vid C:\Users\me\Videos\bug_repro.mp4
> 이 영상에서 버튼 클릭 후 화면이 깨지는데, 원인 코드를 찾아줘
```

### 명령어

| 명령 | 동작 |
|---|---|
| `/img PATH...` | 이미지 첨부 |
| `/vid PATH...` | 동영상 첨부 |
| `/files` | 현재 첨부 목록 |
| `/clear` | 첨부 해제 |
| `/index` | 워크스페이스 재색인 |
| `/notes` | 어시스턴트가 기억 중인 프로젝트 지식 |
| `/reset` | 대화 초기화 (기억은 유지) |
| `/usage` | 토큰 사용량 |
| `/exit` | 종료 |

---

## 동작 구조

```
사용자 입력 (텍스트 + 이미지/동영상)
        │
        ▼
  media/          이미지 리사이즈 · 동영상 → 장면전환 키프레임 추출
        │         (시각 토큰 예산 안에서 프레임 수/해상도 자동 조절)
        ▼
  memory/         BM25로 저장소 검색 → 관련 코드 청크 주입
        │         + 이전 세션에서 기억한 프로젝트 지식
        ▼
  agent/loop.py   ① 계획 수립 (읽기 전용 도구만 허용)
        │         ② 도구 호출 루프 (read/write/edit/grep/shell/python)
        │         ③ 검증: 테스트 실행 → 실패하면 출력 보고 자가 수정
        ▼
  backends/       vLLM · SGLang (OpenAI 호환) 또는 in-process transformers
```

### 동영상 처리가 핵심인 이유

30fps 60초 영상은 1800프레임입니다. 모델이 감당하는 건 32장 정도.
균등 샘플링하면 대부분 정지 화면에 낭비됩니다.

newal은 프레임별 **평활화된 채널 히스토그램**으로 변화량을 점수화해, 실제로 뭔가 바뀐
순간을 골라냅니다. 히스토그램 빈 경계 문제(밝기가 조금만 변해도 만점 처리되는 현상)를
피하려고 채널별로 블러를 겁니다 — 측정값:

| 변화 | 점수 (임계값 0.28) |
|---|---|
| 센서 노이즈 (σ=4) | 0.013 ✅ 무시 |
| 밝기 +5 | 0.138 ✅ 무시 |
| 밝기 +30 | 0.938 ⚠️ 키프레임 |
| 하드 컷 | 1.000 ⚠️ 키프레임 |

3장면 6초 테스트 영상에서 전환 지점(2.0s, 4.0s)을 **오차 0.000초**로 잡아냅니다.

### 시각 토큰 예산

Qwen 비전 인코더는 28×28 패치를 2×2로 병합합니다 → `ceil(W/28)·ceil(H/28)/4` 토큰.
1280×720 프레임 ≈ 300토큰, 32프레임 ≈ 10K토큰. 방치하면 컨텍스트가 터집니다.

`media.visual_token_budget` 안에 들어가도록 **해상도보다 프레임 수를 먼저 줄입니다** —
UI 버그나 에러 텍스트를 읽으려면 흐릿한 32장보다 선명한 8장이 낫기 때문입니다.

---

## 설정

`configs/default.yaml`을 복사해 `configs/local.yaml`로 두면 git에 올라가지 않습니다.
환경변수로도 덮어쓸 수 있습니다:

```powershell
$env:NEWAL_MODEL__ID = "Qwen/Qwen3.5-9B"
$env:NEWAL_BACKEND__MAX_MODEL_LEN = "32768"
```

### 품질을 끌어올리는 설정

| 설정 | 효과 |
|---|---|
| `agent.verify: true` | **가장 중요.** 테스트 자동 실행 + 자가 수정 |
| `agent.verify_command` | 테스트 명령 명시 (미지정 시 프로젝트 구조로 자동 탐지) |
| `agent.max_verify_retries` | 수정 재시도 횟수 (기본 2) |
| `agent.plan_first: true` | 편집 전 계획 수립. 다중 파일 작업 정확도 상승 |
| `model.enable_thinking: true` | thinking 모드. 토큰을 쓰지만 어려운 작업에 효과적 |
| `memory.dense_rerank: true` | BM25 위에 임베딩 재정렬 (`pip install "newal[dense]"`) |

### VRAM이 부족할 때

```yaml
backend:
  max_model_len: 32768      # 256K → 32K (KV 캐시가 VRAM을 가장 많이 먹습니다)
  gpu_memory_utilization: 0.85
  quantization: "fp8"       # 또는 AWQ/GPTQ 4bit 모델 사용
media:
  visual_token_budget: 8192
```

---

## 안전장치

도구는 워크스페이스 밖으로 못 나갑니다:

- 경로 탈출 차단 (`../`, 절대경로, **심볼릭 링크 우회 포함**)
- `denied_paths` (`.git`, `.env`, `.venv` 등) 읽기/쓰기 금지
- `run_shell` / `run_python`은 `tools.shell_policy`로 통제 — `ask`(기본) / `allow` / `deny`
- `allow`여도 파괴적 명령(`rm -rf`, `mkfs`, `dd of=/dev/`, fork bomb, `git push --force` 등)은 무조건 거부
- `edit_file`은 매칭이 모호하면 실패 — 조용한 오편집 방지

---

## 개발

```powershell
python -m pytest -q          # 61개 테스트
python -m newal index        # 저장소 색인만
```

```
src/newal/
├─ config.py          설정 로딩 (YAML 레이어 + 환경변수)
├─ cli.py             대화형 터미널
├─ backends/          vLLM/SGLang 클라이언트 · 서버 자동 기동 · transformers 폴백
├─ media/             이미지/동영상 → 콘텐츠 파트, 시각 토큰 예산
├─ memory/            SQLite 저장소 · BM25 · 증분 색인
└─ agent/             도구 · 프롬프트 · 실행 루프 · 검증기
```

`AGENTS.md` / `CLAUDE.md` / `.newal.md`가 저장소에 있으면 시스템 프롬프트에 자동 반영됩니다.

---

## 라이선스

MIT. Qwen 가중치는 각 모델의 라이선스를 따릅니다.

# newal

**Qwen 모델 풀 기반 로컬 멀티모달 코딩 어시스턴트** — 텍스트 · 이미지 · 동영상 입력을 받아 내 PC에서만 동작합니다.

Python · Visual Studio 2026 · 완전 오프라인 · 사용량 제한 없음 · 무료.

**v1.0** — 단일 모델에서 **이종(heterogeneous) 모델 풀 + 라우터** 구조로 전환.

---

## 먼저: "여러 모델 쓰면 더 똑똑해진다"는 절반만 맞습니다

멀티모델이 유행이지만, 2026년 실측 연구는 생각보다 냉정합니다.

67개 프런티어 모델을 대상으로 라우팅·투표·Mixture-of-Agents를 측정한
[co-failure ceiling 연구](https://arxiv.org/abs/2606.27288)의 결론:

> 앙상블 정확도는 **1 − β**를 넘을 수 없다. β는 *모든 모델이 같은 문제에서 동시에 틀리는 비율*.
> 이득은 "모델을 더 넣어서"가 아니라 **"서로 다른 문제에서 틀려야"** 나온다.

그리고 실측 co-failure는 이론 추정치보다 **약 2.5배 높았습니다** (수학 0.052 vs 예측 0.023,
코드 실행 0.079). 즉 모델들은 생각보다 **같이 틀립니다.**

**이게 왜 우리한테 치명적이냐면** — Qwen3.5-4B + Qwen3.5-9B + Qwen3.6-35B를 모아 투표시키는 건
사실상 **Self-MoA**입니다. 같은 사전학습 데이터, 같은 사각지대. 같은 문제에서 사이좋게 틀립니다.
VRAM 3배 쓰고 얻는 게 거의 없습니다. 같은 연구가 명시적으로 말합니다 — *"low-correlation
diverse ensembles outperform high-correlation Self-MoA"*.

**그래서 newal은 투표 앙상블을 만들지 않았습니다.** 대신 **서로 다른 일을 하는 전문가 풀**을
구성했습니다. 하는 일이 다르면 실패도 구조적으로 독립적입니다.

| 멤버 | 하는 일 | 왜 상관관계가 낮은가 |
|---|---|---|
| `heavy` | 계획 · 코딩 · 수리 | — (기준) |
| `light` | 기계적 편집 · 요약 | 같은 일을 함 → **정확도 이득 없음. 속도/VRAM 목적** |
| `embedding` | 밀집 검색 | 생성이 아닌 검색 → 실패 방식이 다름 |
| `reranker` | 교차 인코더 재정렬 | 생성이 아닌 순위 → 실패 방식이 다름 |
| `draft` | 추측 디코딩 | **출력 분포 동일** → 품질 변화 0, 속도만 |
| **검증기(테스트)** | 실행 결과 | **모델이 아님 → β ≈ 0. 진짜 이득의 원천** |

마지막 줄이 핵심입니다. 이 시스템에서 모델 오류와 **상관관계가 없는 유일한 신호는
"코드를 실제로 실행한 결과"** 입니다. 그래서 에스컬레이션은 *모델끼리의 의견 불일치*가 아니라
**실행 실패**로 트리거됩니다.

### Codex / Claude Code 대비

| 축 | newal | Codex / Claude Code |
|---|---|---|
| **동영상 입력** | 네이티브. 장면 전환 프레임 추출 | 미지원 |
| **이미지 입력** | 네이티브 | 지원 |
| **영구 프로젝트 기억** | SQLite, 세션 간 유지 | 세션 종료 시 소멸 |
| **검증 루프** | 테스트 실행 → 실패 분석 → 자가 수정 | 부분적 |
| **하이브리드 검색** | BM25 + 밀집 + 재정렬 | 부분적 |
| **비용 / 제한** | 0원, 무제한, 오프라인 | 종량 과금, 레이트 리밋 |
| **원시 추론 능력** | **약함** | 강함 |

로컬 35B-A3B가 프런티어 모델의 원시 추론을 이기지는 못합니다. 전략은
**"덜 똑똑한 모델을 더 좋은 루프에 넣기"** 입니다.

---

## 모델 구성

Qwen3.5부터 **별도의 `-VL` 계열이 없습니다.** 본체 모델이 텍스트+이미지+동영상을 네이티브로
처리합니다(early-fusion). `Qwen3-VL`은 구세대입니다.

```yaml
models:
  heavy:                                  # 기본값. 이것만 켜져 있음
    id: "Qwen/Qwen3.6-35B-A3B"            # 35B/3B MoE, 멀티모달, 256K
    tier: 3
    speculative_draft: "Qwen/Qwen3.5-0.8B"  # 추측 디코딩 (무손실 가속)
  light:
    id: "Qwen/Qwen3.5-4B"                 # 기계적 작업 전담
    tier: 1
    enabled: false                        # GPU 여유 생기면 켜기
  embedding:
    id: "Qwen/Qwen3-Embedding-0.6B"
    enabled: false
  reranker:
    id: "Qwen/Qwen3-Reranker-0.6B"
    enabled: false
```

**기본값은 `heavy` 하나만 켜져 있습니다.** 네 개를 다 켜면 대부분의 개인 PC에서 OOM 납니다.
VRAM 여유를 보고 하나씩 켜세요. 멤버마다 별도 서버 프로세스입니다.

### 켜는 순서 추천 (효과 대비 VRAM)

1. **`embedding`** (~1GB) — 검색 품질이 가장 크게 오릅니다. 상관관계 낮은 진짜 이득.
2. **`reranker`** (~1GB) — 검색 정밀도 추가 상승.
3. **`speculative_draft`** (~1GB) — 속도만. 품질 변화 없음.
4. **`light`** (~3GB) — 지연시간/VRAM 최적화. **정확도는 안 오릅니다.**

### 하드웨어

| 모델 | 총/활성 | 4bit VRAM | 권장 GPU |
|---|---|---|---|
| **`Qwen/Qwen3.6-35B-A3B`** | 35B / 3B MoE | ~20GB | RTX 4090·5090 |
| `Qwen/Qwen3.5-27B` | 27B dense | ~16GB | RTX 4090 |
| `Qwen/Qwen3.5-9B` | 9B dense | ~6GB | RTX 4070·4080 |
| `Qwen/Qwen3.5-4B` | 4B dense | ~3GB | RTX 3060 12GB |

전부 256K 컨텍스트(YaRN으로 1M 확장 가능), thinking 모드 내장.

> **왜 35B-A3B가 기본값인가:** MoE라 토큰당 3B만 활성화됩니다. 27B dense와 VRAM은 비슷한데
> 생성이 몇 배 빠릅니다. 에이전트 루프는 한 턴에 호출을 수십 번 하므로 속도가 곧 체감 품질입니다.

---

## 라우팅: 호출마다 두 가지를 결정

매 호출 전에 난이도 점수(0~1)를 계산해 **어느 티어로 보낼지**와 **thinking을 켤지**를 정합니다.
[캐스케이드 라우팅](https://arxiv.org/pdf/2606.27457) 및
[inhibitory deliberation](https://arxiv.org/pdf/2606.06745) 계열 접근입니다.

점수는 학습된 라우터가 아니라 **투명한 휴리스틱**입니다 — 검사 가능하고, 결정적이고,
추가 지연이 없고, 사용자가 갖고 있지 않은 학습 데이터를 요구하지 않습니다.

```
역할 기준점        triage 0.00 · edit 0.20 · vision 0.50 · code 0.55 · plan 0.75 · repair 0.90
+ 실행 증거        검증 실패 +0.40 · 에스컬레이션 +0.15/회 · 도구 오류 +0.07/건
+ 요청 형태        조사형 표현 +0.20 · 기계적 표현 −0.20 · 긴 요청 +0.08~0.15
+ 첨부             동영상 +0.15 · 이미지 +0.10 · 다중 파일 +0.12
```

실행 증거가 요청 형태보다 가중치가 큽니다. 추측이 아니라 관측이기 때문입니다.

한국어 표현도 인식합니다 (한국어 동사 활용까지 — `바꾸`/`바꿔`/`변경` 모두 매칭):

```
변수 이름 바꿔줘      → light, thinking off, 0.35
오타 고쳐줘           → light, thinking off, 0.35
이 버그 원인이 뭐야?   → heavy, thinking on,  0.75
아키텍처 다시 설계하자 → heavy, thinking on,  0.75
```

`plan`과 `repair`는 점수와 무관하게 **절대 싼 티어로 안 갑니다.** 잘못된 계획과 잘못된 수리는
아낀 토큰보다 비쌉니다.

### 실제 동작 예시

```
$ 변수 이름 바꿔줘
⇢ light (code, direct, score=0.35)      # 기계적 요청 → 싼 티어
⇢ light (code, direct, score=0.35)
⇢ heavy (code, thinking, score=0.60)    # 도구 오류 → 에스컬레이션
· verifying
· verify: failed (repair attempt 1)     # 테스트 실패 → 실행 증거
⇢ heavy (repair, thinking, score=0.80)  # repair 역할, 최상위 티어 고정
· verify: passed after 1 repair attempt(s)

heavy: 74 tok / light: 30 tok / escalated 1x
```

---

## 검색: 3단 하이브리드

각 단계는 **모델이 없으면 조용히 건너뜁니다.** 임베딩 서버가 죽어도 세션은 안 죽고 BM25로 답합니다.

```
1. BM25          항상 동작. 모델 불필요
2. 밀집 검색      Qwen3-Embedding → 코사인 유사도
   └ 융합         Reciprocal Rank Fusion
3. 재정렬         Qwen3-Reranker 교차 인코더
```

**왜 RRF인가:** BM25 점수는 무한대 범위에 코퍼스 의존적이고, 코사인은 [-1, 1]입니다. 원점수를
가중합하면 가중치가 무의미해집니다. RRF는 **점수가 아니라 순위**를 합치므로 스케일 보정이
필요 없습니다.

---

## 설치 (Windows + Visual Studio 2026)

### 1. 프로젝트 열기

```
File → Open → Project/Solution → newal.sln
```
폴더 모드로 열어도 `launch.vs.json` 실행 구성이 잡힙니다.

### 2. 가상환경 + 의존성

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install vllm          # 또는 pip install "sglang[all]"
```

> **Windows에서 vLLM 설치가 깨질 때:** WSL2를 쓰거나 `configs/local.yaml`에
> `runtime: {kind: transformers}`로 폴백하세요 (느리지만 서버 불필요).

### 3. 실행

VS 실행 버튼 드롭다운: **newal: chat** / **newal: start Qwen servers** / **newal: run tests**

```powershell
python -m newal chat
```

첫 실행 시 가중치를 자동으로 내려받습니다 (수십 GB, 시간 걸림).

---

## 사용법

```
> UserService의 인증 로직을 리팩터링해줘

> /img C:\Users\me\Desktop\error.png
> 이 에러 화면 원인이 뭐야?

> /vid C:\Users\me\Videos\bug_repro.mp4
> 이 영상에서 버튼 클릭 후 화면이 깨지는데, 원인 코드를 찾아줘
```

| 명령 | 동작 |
|---|---|
| `/img PATH...` / `/vid PATH...` | 이미지 / 동영상 첨부 |
| `/files` `/clear` | 첨부 목록 / 해제 |
| `/index` | 워크스페이스 재색인 |
| `/models` | **모델 풀 + 모델별 토큰 사용량** |
| `/notes` | 기억 중인 프로젝트 지식 |
| `/reset` `/usage` `/help` `/exit` | — |

---

## 동영상 처리

30fps 60초 영상은 1800프레임, 모델이 감당하는 건 32장. 균등 샘플링하면 대부분 정지 화면에
낭비됩니다. newal은 **평활화된 채널 히스토그램**으로 변화량을 점수화해 실제로 뭔가 바뀐
순간을 고릅니다.

| 변화 | 점수 (임계값 0.28) |
|---|---|
| 센서 노이즈 σ=4 | 0.013 ✅ 무시 |
| 밝기 +5 | 0.138 ✅ 무시 |
| 밝기 +30 | 0.938 ⚠️ 키프레임 |
| 하드 컷 | 1.000 ⚠️ 키프레임 |

3장면 6초 테스트 영상에서 전환 지점(2.0s, 4.0s)을 **오차 0.000초**로 잡습니다.

**시각 토큰 예산:** Qwen 비전 인코더는 28×28 패치를 2×2 병합 → `ceil(W/28)·ceil(H/28)/4` 토큰.
`media.visual_token_budget` 안에 들어가도록 **해상도보다 프레임 수를 먼저 줄입니다** —
UI 버그나 에러 텍스트를 읽으려면 흐릿한 32장보다 선명한 8장이 낫기 때문입니다.

---

## 설정

`configs/default.yaml` → `configs/local.yaml`로 복사하면 git에 안 올라갑니다.

```powershell
$env:NEWAL_MODELS__LIGHT__ENABLED = "true"
$env:NEWAL_ROUTER__STRATEGY = "single"
```

### 품질을 끌어올리는 설정

| 설정 | 효과 |
|---|---|
| `agent.verify: true` | **가장 중요.** 실행 증거 = 유일한 무상관 신호 |
| `agent.verify_command` | 테스트 명령 명시 (미지정 시 자동 탐지) |
| `models.embedding.enabled` | 검색 품질. 상관관계 낮은 진짜 이득 |
| `models.reranker.enabled` | 검색 정밀도 추가 상승 |
| `router.strategy: cascade` | 지연시간 절감 (정확도는 아님) |
| `router.explain: true` | 라우팅 결정과 근거를 매 호출 출력 |
| `router.thinking.mode: adaptive` | 쉬운 작업에서 thinking 토큰 절약 |

### VRAM이 부족할 때

```yaml
runtime:
  max_model_len: 32768      # 256K → 32K (KV 캐시가 VRAM을 가장 많이 먹음)
  quantization: "fp8"
models:
  heavy:
    speculative_draft: null # 드래프트 모델 끄기
media:
  visual_token_budget: 8192
```

---

## 안전장치

- 경로 탈출 차단 (`../`, 절대경로, **심볼릭 링크 우회 포함**)
- `denied_paths` (`.git`, `.env`, `.venv` 등) 읽기/쓰기 금지
- `run_shell` / `run_python`은 `tools.shell_policy`로 통제 — `ask`(기본) / `allow` / `deny`
- `allow`여도 파괴적 명령(`rm -rf`, `mkfs`, `dd of=/dev/`, fork bomb, `git push --force`)은 무조건 거부
- `edit_file`은 매칭이 모호하면 실패 — 조용한 오편집 방지

---

## 개발

```powershell
python -m pytest -q          # 126개 테스트
python -m newal index        # 저장소 색인만
```

```
src/newal/
├─ config.py          설정 로딩 (YAML 레이어 + 환경변수)
├─ cli.py             대화형 터미널
├─ models/            ★ 모델 풀 · 라우터 · 역할 · 임베딩/재정렬 클라이언트
├─ backends/          vLLM/SGLang 클라이언트 · 서버 기동 · transformers 폴백
├─ media/             이미지/동영상 → 콘텐츠 파트, 시각 토큰 예산
├─ memory/            SQLite · BM25 · RRF 융합 · 증분 색인
└─ agent/             도구 · 프롬프트 · 실행 루프 · 검증기
```

`AGENTS.md` / `CLAUDE.md` / `.newal.md`가 있으면 시스템 프롬프트에 자동 반영됩니다.

### 버전 정책

- **1.0** — 이종 모델 풀 + 라우터 (현재)
- **0.1** — 단일 모델 + 검증 루프

기능/모델 업데이트는 major, 버그 수정은 minor로 올립니다.

---

## 참고 문헌

- [When Does Combining Language Models Help? A Co-Failure Ceiling](https://arxiv.org/abs/2606.27288) — 앙상블 상한, Self-MoA의 한계
- [Cluster, Route, Escalate: Cost-Aware LLM Serving](https://arxiv.org/pdf/2606.27457) — 캐스케이드 라우팅
- [UCCI: Calibrated Uncertainty for Cost-Optimal Cascade Routing](https://arxiv.org/pdf/2605.18796)
- [When to Think Deeply: Inhibitory Deliberation](https://arxiv.org/pdf/2606.06745) — 적응형 thinking
- [Qwen3 Embedding & Reranker](https://qwenlm.github.io/blog/qwen3-embedding/)

## 라이선스

MIT. Qwen 가중치는 각 모델의 라이선스를 따릅니다.

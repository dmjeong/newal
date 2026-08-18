# newal

**Qwen 모델 풀 기반 로컬 멀티모달 코딩 어시스턴트** — 텍스트 · 이미지 · 동영상 입력을 받아 내 PC에서만 동작합니다.

Python · VS Code · 완전 오프라인 · 사용량 제한 없음 · 무료.

**v5.0** — 웹 UI가 두 모드로 갈라졌습니다. `/`는 평소 쓰는 대화 화면과 개인 설정,
`/training`은 파인튜닝 준비 전용입니다. **VS Code에서 F5를 누르면 웹이 뜹니다.**

---

## 빠른 시작

```bash
git clone https://github.com/dmjeong/newal && cd newal
code .                        # VS Code로 열기
```

**1단계.** Ctrl+Shift+P → `Tasks: Run Task` → **`setup: create venv and install`**
가상환경 생성과 설치가 한 번에 끝납니다.

**2단계.** **F5.** 브라우저가 열리고 http://localhost:8800 이 뜹니다.

F5는 목록 맨 위 구성을 실행합니다. 지금은 `newal: web (browser UI)`입니다.
다른 걸 돌리고 싶으면 Ctrl+Shift+D의 드롭다운에서 고르세요:

| 구성 | 하는 일 |
|---|---|
| **`newal: web (browser UI)`** | **F5 기본값.** 모델 서버를 띄우고 브라우저를 엽니다 |
| `newal: web without starting servers` | UI만. **GPU도 모델도 없이** 화면이 뜹니다 |
| `newal: chat (interactive)` | 터미널 대화 |
| `newal: run all tests` | 전체 테스트 |

터미널로 하려면:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,video,web]"
pytest -q                          # 299개 통과하면 설치 성공 (GPU/모델 불필요)
newal web
```

### GPU가 없으면 어디까지 되나

`newal web`은 모델 서버부터 띄우려 하고, GPU가 없으면 거기서 막힙니다.
**`--no-autostart`를 붙이면 UI는 그대로 뜹니다:**

```bash
newal web --no-autostart
```

모델이 필요한 건 **대화 한 턴뿐**입니다. 나머지는 다 됩니다 — 두 화면, 설정 저장,
학습 데이터 내보내기, 스크립트 생성. 사이드바가 어느 모델에 연결 못 했는지 알려주고,
대화창은 물어보면 그때 이유를 말합니다.

실제로 대화하려면 추론 엔진이 필요합니다:

```bash
pip install vllm                   # NVIDIA GPU 필요
newal web                          # 첫 실행 시 가중치 자동 다운로드 (수십 GB)
```

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

1. **`embedding`** (~1GB) — 검색 품질 + **semantic 라우팅**. 켜기 전엔 `auto`가 llm 단계로만 동작합니다.
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

## 라우팅: 질문 보고 알아서 고릅니다

매 호출 전에 **어느 모델로 보낼지**와 **thinking을 켤지**를 정합니다.
[캐스케이드 라우팅](https://arxiv.org/pdf/2606.27457),
[inhibitory deliberation](https://arxiv.org/pdf/2606.06745) 계열입니다.

핵심 아이디어: **결정 자체도 캐스케이드합니다.** 대부분의 질문은 명백히 쉽거나 명백히 어렵습니다.
그런 건 공짜 휴리스틱으로 끝냅니다. 애매한 소수만 진짜 분류기를 씁니다.

```
1. 휴리스틱     난이도 점수 계산. 0원, 0ms, 결정적
      │
      │ 점수가 임계값 ±0.12 밖 → 여기서 끝 (대부분의 질문)
      │ 점수가 애매한 구간 안 → 아래로
      ▼
2. semantic     쿼리 임베딩 → 라벨된 예시와 kNN 투표. ~20ms
      │
      │ 확신 부족(<0.3) → 아래로
      ▼
3. llm          싼 모델에게 한 단어로 물어봄. 짧은 호출 1회
```

`router.mode`로 고를 수 있습니다: `heuristic` / `semantic` / `llm` / **`auto`(기본, 위 3단)**.
`uncertainty_band: 0` 으로 두면 분류기를 완전히 끕니다.

### 왜 휴리스틱을 안 버렸나

정규식은 제가 적어둔 표현만 압니다. 확장이 안 됩니다 — 맞는 지적이었습니다.
하지만 **버리는 대신 1차 필터로 남겼습니다.** "오타 고쳐줘"에 임베딩 호출을 태울 이유가 없습니다.
분류기는 휴리스틱이 스스로 자신 없는 구간에서만 돕니다.

난이도 점수:

```
역할 기준점        triage 0.00 · edit 0.20 · vision 0.50 · code 0.55 · plan 0.75 · repair 0.90
+ 실행 증거        검증 실패 +0.40 · 에스컬레이션 +0.15/회 · 도구 오류 +0.07/건
+ 요청 형태        조사형 +0.20 · 기계적 −0.20 · 긴 요청 +0.08~0.15
+ 첨부             동영상 +0.15 · 이미지 +0.10 · 다중 파일 +0.12
```

실행 증거가 요청 형태보다 가중치가 큽니다. 추측이 아니라 관측이기 때문입니다.
`plan`과 `repair`는 점수와 무관하게 **절대 싼 티어로 안 갑니다.**

### 내 저장소에서 배웁니다 ★

여기가 GPT/Claude 라우터가 못 하는 부분입니다. 매 턴이 끝나면 **실제로 무슨 일이 있었는지**를
라벨로 저장합니다:

| 처음 보낸 곳 | 결과 | 배우는 것 |
|---|---|---|
| 싼 모델 | 에스컬레이션 or 테스트 실패 | → `strong` (강한 모델이 필요했다) |
| 싼 모델 | 깔끔하게 끝남 | → `cheap` (싼 걸로 충분했다) |
| 강한 모델 | 깔끔하게 끝남 | **라벨 안 함** |

마지막 줄이 중요합니다. 강한 모델이 잘 끝냈다고 해서 *싼 모델로도 됐을지*는 알 수 없습니다.
여기서 `strong`을 기록하면 예시 집합이 시간이 갈수록 `strong` 쪽으로 편향됩니다. 그래서
**정보가 실제로 있는 결과만** 기록합니다.

라벨의 출처가 **내 저장소에서 코드를 실제로 실행한 결과**라는 게 핵심입니다. 호스팅 라우터는
내 코드베이스에서 어떤 요청이 어려운지 알 수가 없습니다.

실측: "이 함수 동작이 이상한데 좀 봐줘"가 한 번 실패하고 나면, 다음에 "이 함수 동작이 좀 이상해"가
`strong`(확신 0.61)으로 분류됩니다. 처음엔 아무 마커도 없어서 휴리스틱이 못 잡던 표현입니다.

`/routing` 으로 현재 설정과 학습된 내역을 볼 수 있습니다.

### 실제 동작 예시

```
$ 변수 이름 바꿔줘
⇢ light (code, direct, score=0.35)      # 기계적 → 싼 티어 (휴리스틱만, 분류기 안 씀)
⇢ heavy (code, thinking, score=0.60)    # 도구 오류 → 에스컬레이션
· verify: failed (repair attempt 1)     # 테스트 실패 = 실행 증거
⇢ heavy (repair, thinking, score=0.80)  # repair는 최상위 고정
· verify: passed

heavy: 74 tok / light: 30 tok / escalated 1x
learned: strong <- '변수 이름 바꿔줘'    ← 다음엔 처음부터 heavy로
```

### 분류기가 틀릴 때의 안전장치

- 확신이 `min_classifier_confidence`(0.3) 미만이면 **무시하고 휴리스틱을 씁니다**
- 예시 중 닮은 게 하나도 없으면 **기권**합니다 (0점짜리 라벨을 지어내지 않음)
- 임베딩/분류 모델이 죽어도 **휴리스틱으로 조용히 내려앉습니다**
- 어떤 경우에도 라우팅 실패가 세션을 죽이지 않습니다

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

## 설치 상세

### 1. VS Code로 열기

```bash
code .
```

`.vscode/`에 실행 구성이 커밋되어 있어서 폴더만 열면 바로 잡힙니다.
처음 열면 권장 확장(Python, Pylance, Ruff) 설치를 물어봅니다 — 설치하세요.
Pylance가 있어야 `src/` 레이아웃에서 정의로 이동(F12)이 동작합니다.

### 2. 의존성

**Ctrl+Shift+P → Tasks: Run Task** 에서:

| 태스크 | 언제 |
|---|---|
| `setup: create venv and install` | 클론 직후 한 번 |
| `setup: install the vLLM inference engine` | 실제로 모델을 돌릴 때 (용량 큼) |
| `test` | 아무 때나. GPU 불필요 |
| `index the workspace` | 검색 인덱스만 다시 만들기 |
| `show the resolved config` | 설정이 왜 저렇게 먹었는지 확인 |

인터프리터가 안 잡히면 **Ctrl+Shift+P → Python: Select Interpreter → `.venv`**.

### 3. 실행 (F5)

**Ctrl+Shift+D**(Run and Debug 패널)에서 고르거나 **F5**:

| 구성 | 하는 일 |
|---|---|
| `newal: chat (interactive)` | 평소 쓰는 것 |
| `newal: chat (verbose, ...)` | 로그 켜고 실행. 뭔가 안 될 때 |
| `newal: chat without starting servers` | 서버를 따로 띄워놨을 때 |
| `newal: start model servers only` | 모델 서버만 |
| `newal: index the workspace` | 색인만 |
| `newal: run all tests` / `run the open test file` | 테스트 |

전부 중단점이 걸립니다. 라우팅이 왜 저 모델을 골랐는지 보려면
`src/newal/models/router.py`의 `route()`에 중단점을 걸고 F5를 누르세요.

터미널만 쓸 거면:

```bash
newal chat            # 대화
newal serve           # 모델 서버만
newal index           # 색인만
newal config          # 병합된 설정 출력
newal config -s router  # 한 섹션만
```

> **Windows에서 vLLM 설치가 깨질 때:** WSL2를 쓰거나 `configs/local.yaml`에
> `runtime: {kind: transformers}`로 폴백하세요 (느리지만 서버 불필요).

---

## 브라우저 UI

```bash
newal web                    # http://localhost:8800, 브라우저 자동 실행
newal web --port 9000
newal web --no-browser
```

**빌드 스텝이 없습니다.** UI는 패키지에 동봉된 순수 HTML/CSS/JS라 npm도 Node도
필요 없습니다. `pip install`이 전부입니다.

주소가 두 개입니다. 하는 일도, 위험도도 다르기 때문에 한 화면에 섞지 않았습니다.

| 주소 | 모드 | 하는 일 |
|---|---|---|
| `/` | 일반 사용 | 대화, 첨부, 셸 승인, 개인 설정 |
| `/training` | 학습 | 수집 현황, 데이터셋 내려받기, 파인튜닝 파라미터, 스크립트 생성 |

### `/` — 일반 사용 모드

- 스트리밍은 SSE. 계획·라우팅·도구 호출·검증이 진행되는 대로 보입니다
- 이미지·동영상은 **드래그앤드롭**
- `shell_policy: ask`면 셸 실행 전에 다이얼로그로 물어봅니다. 탭을 닫거나
  Esc를 누르면 **거부**로 처리됩니다
- 사이드바에 모델 풀, 라우팅 상태, 토큰 사용량, 수집된 학습 데이터
- 다크 모드는 OS 설정을 따릅니다

**설정** 버튼을 누르면 자주 만지는 값 23개가 나옵니다. 에이전트 동작(계획 · 검증 ·
수리 횟수), 라우팅(전략 · 임계값 · thinking), 생성(temperature · top_p), 미디어
(프레임 수 · 해상도), 안전(`shell_policy` · 타임아웃), 기록(수집 · 트랜스크립트).

바꾸면 곧바로 적용되고 `configs/local.yaml`에 **바뀐 값만** 저장됩니다. 나머지
기본값은 파일에 박히지 않으니 다음 버전에서 기본값이 좋아지면 그대로 따라갑니다.
값 하나라도 범위를 벗어나면 전부 되돌립니다. 반쯤 적용된 설정이 조용히 남는 쪽이
더 나쁩니다.

모델 풀 · 포트 · DB 경로는 여기 없습니다. 시작할 때 한 번 읽고 마는 값이라
바꿔봐야 화면만 바뀌고 실제로는 안 바뀝니다. 그건 `configs/local.yaml`을 직접
고치고 재시작하세요.

### `/training` — 학습 모드

- 수집된 턴·수리 쌍·라우팅 결과 개수
- SFT / DPO / 라우팅 데이터셋 JSONL 내려받기 (없으면 빈 파일 대신 안내를 띄웁니다)
- 파인튜닝 파라미터: task · 베이스 모델 · QLoRA/LoRA/full · LoRA r/alpha/dropout ·
  target modules · lr · epochs · 배치 · grad accum · max seq length · warmup · DPO beta
- **바로 실행되는 TRL 스크립트**를 만들어 줍니다

newal은 **준비까지만** 합니다. 학습은 GPU로 몇 시간 걸리는 작업이라 브라우저 요청
안에서 돌릴 일이 아닙니다. 데이터셋과 스크립트를 받아 터미널에서 돌리세요.

```bash
pip install trl peft bitsandbytes datasets
python train_sft.py
```

막을 값과 말릴 값은 구분합니다. `lora_r: 0`처럼 돌아갈 수 없는 값은 422로
거부하고, `alpha = r`처럼 흔치 않지만 가능한 값은 경고만 띄우고 넘어갑니다.
샘플이 200개 밑이면 그것도 말해 줍니다.

> **기본은 127.0.0.1 바인딩입니다.** 이 UI에는 인증이 없고, 에이전트는 파일을
> 수정하며 `shell_policy`에 따라 명령도 실행합니다. `--host`로 외부에 열면
> 접근 가능한 사람이 그 권한을 그대로 가집니다. `/training`에는 **전부 삭제**
> 버튼도 있습니다.

윈도우 응용프로그램은 만들 필요가 없습니다. 나중에 네이티브 창이 필요해지면
같은 UI를 pywebview로 감싸면 됩니다.

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
| `/models` | 모델 풀 + 모델별 토큰 사용량 |
| `/routing` | **라우팅 설정 + 학습된 내역** |
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

기본값은 패키지 안에 있습니다 (`src/newal/data/default.yaml`). 바꾸려면
`configs/local.yaml`을 만드세요. gitignore 되어 있고, 기본값 위에 병합됩니다.

```bash
newal config > configs/local.yaml   # 현재 설정을 뽑아서 시작
```

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
| `router.mode: auto` | 애매한 질문에서 학습된 분류기가 판단 |
| `router.learn_from_outcomes` | 실행 결과로 라우팅이 계속 좋아짐 |
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

## 쓰면 쓸수록 학습 데이터가 쌓입니다 ★

파인튜닝의 진짜 병목은 프레임워크가 아니라 **라벨 붙은 도메인 데이터**입니다.
newal은 그걸 부산물로 만들어냅니다 — 매 턴이 진짜 저장소에 대한 진짜 요청이고,
검증 루프가 **테스트를 실제로 돌려서** 객관적인 라벨을 붙이기 때문입니다.

**라벨링 경로에 사람도, LLM judge도 없습니다.** 전부 실행 결과입니다.

```
한 번에 통과한 턴          → SFT 샘플
검증 실패 후 수리된 턴      → DPO 선호 쌍 (실패한 편집 = rejected, 통과한 수리 = chosen)
라우팅 결과                → 분류 데이터셋
```

```bash
newal export                                        # 뭐가 쌓였는지 확인
newal export --format sft     --out sft.jsonl       # {"messages": [...]}
newal export --format dpo     --out dpo.jsonl       # {"prompt","chosen","rejected"}
newal export --format routing --out routing.jsonl   # {"text","label"}
newal export --clear                                # 전부 삭제 (확인 후)
```

전부 TRL 트레이너가 바로 먹는 형식입니다.

### 데이터 품질을 위해 걸러내는 것

| 제외 대상 | 이유 |
|---|---|
| **수리된 턴을 SFT에서 제외** | 히스토리에 실패한 시도가 그대로 남아 있어, 학습하면 **틀린 뒤 고치는 습관**을 배웁니다. 이런 턴은 DPO 쌍으로만 씁니다 |
| **첨부가 있던 턴 제외** | 이미지는 리댁션돼 사라졌는데 대화는 그림 얘기를 합니다. 학습하면 **못 본 그림을 아는 척**하게 됩니다 |
| **검증 안 된 턴 제외** | 기본값. `--include-unverified`로 포함 가능 |

### 첫 파인튜닝은 `light`부터

`heavy`를 튜닝할 이유가 없습니다. 작은 모델에게 *이 저장소의 관습*을 가르쳐서
에스컬레이션 없이 기계적 작업을 처리하게 만드는 게 목표입니다.

**성공 기준이 이미 측정됩니다** — `newal export`의 라우팅 카운트에서 `strong` 비율이
떨어지면 효과가 있는 겁니다. 별도 벤치마크 없이 실사용에서 나옵니다.

> **개인정보:** 수집된 턴에는 **대화에 등장한 소스 코드가 그대로** 들어갑니다.
> 전부 로컬(`.newal/memory.db`)이고 어디로도 전송되지 않지만, 민감하게 다루세요.
> 첨부(이미지·동영상)는 항상 리댁션됩니다. 끄려면 `training.enabled: false`.

---

## 세션 기록

매 턴이 `.newal/transcripts/session-<타임스탬프>.jsonl`에 한 줄씩 쌓입니다 —
질문, 답변, 어느 모델이 처리했는지, 뭘 고쳤는지, 검증 결과.

**이미지·동영상 프레임은 기록 전에 제거됩니다.** 경로만 남습니다. 동영상 한 턴이
base64 프레임 수십 장이라 그대로 쓰면 턴당 수 MB씩 불어나고, 사용자 스크린샷이
예상치 못한 위치에 저장됩니다.

다만 **대화에 등장한 소스 코드는 그대로 남습니다.** 이 디렉터리는 민감하게 다루세요
(`.gitignore`에 `.newal/`이 이미 들어 있습니다). 끄려면 `ui.save_transcripts: false`.

---

## 안전장치

- 경로 탈출 차단 (`../`, 절대경로, **심볼릭 링크 우회 포함**)
- `denied_paths` (`.git`, `.env`, `.venv` 등) 읽기/쓰기 금지
- `run_shell` / `run_python`은 `tools.shell_policy`로 통제 — `ask`(기본) / `allow` / `deny`
- `allow`여도 파괴적 명령(`rm -rf`, `mkfs`, `dd of=/dev/`, fork bomb, `git push --force`)은 무조건 거부
- `edit_file`은 매칭이 모호하면 실패 — 조용한 오편집 방지

---

## 코드 읽는 법

5,000줄 정도라 한 번에 보면 막막합니다. **읽는 순서**를 추천합니다.

### 한 번의 대화가 지나가는 경로

```
cli.py                사용자 입력 받기, /명령 처리
  └─ agent/loop.py    ★ 여기가 심장. 아래를 순서대로 호출
       ├─ models/router.py      어느 모델로 보낼지 결정
       ├─ backends/…            결정된 모델 호출
       ├─ agent/tools.py        모델이 요청한 도구 실행 (파일 읽기/수정/셸)
       └─ agent/verifier.py     테스트 돌려서 검증, 실패하면 위로 되돌림
```

**`agent/loop.py`의 `Agent.run()` 하나만 읽어도 전체 구조가 잡힙니다.** (약 40줄)

### 파일별 역할

| 파일 | 줄 | 역할 |
|---|---|---|
| **`agent/loop.py`** | 515 | ★ 실행 루프. 계획 → 도구 → 검증 → 수리 |
| `agent/tools.py` | 485 | 모델이 쓰는 도구 + 워크스페이스 샌드박스 |
| `cli.py` | 403 | 터미널 UI, `/명령` |
| `config.py` | 336 | 설정 4계층 병합 |
| `memory/store.py` | 326 | SQLite — 청크 · 노트 · 라우팅 학습 데이터 |
| `memory/indexer.py` | 324 | 색인 + 3단 하이브리드 검색 |
| `media/video.py` | 291 | ★ 동영상 → 장면 전환 키프레임 |
| `models/classifier.py` | 270 | ★ 학습된 쿼리 분류 (semantic / llm) |
| `models/router.py` | 269 | ★ 난이도 점수 + 티어 결정 |
| `backends/launcher.py` | 258 | vLLM/SGLang 서버 기동 |
| `models/pool.py` | 198 | 모델 풀 생명주기 |
| `backends/openai_compat.py` | 182 | 모델 호출 (thinking 스위치 처리) |
| `models/retrieval.py` | 151 | 임베딩 · 재정렬 클라이언트 |
| `agent/prompts.py` | 100 | 시스템 프롬프트 |
| `memory/bm25.py` | 98 | BM25 (외부 의존성 없이 직접 구현) |
| `media/budget.py` | 90 | 시각 토큰 예산 계산 |
| `agent/verifier.py` | 89 | 테스트 명령 탐지 + 실행 |
| `memory/fusion.py` | 62 | RRF 순위 융합 |
| `models/roles.py` | 47 | 역할 정의 + 난이도 기준점 |

★ = 이 프로젝트에서 실제로 재미있는 부분

### 관심사별 진입점

| 알고 싶은 것 | 볼 파일 |
|---|---|
| 모델을 어떻게 고르나 | `models/router.py` → `difficulty_score()`, `Router.route()` |
| 자동 선택이 어떻게 학습되나 | `models/classifier.py` → `label_from_outcome()` |
| 동영상에서 프레임을 어떻게 고르나 | `media/video.py` → `select_keyframes()` |
| 검색이 어떻게 되나 | `memory/indexer.py` → `RepoIndex.search()` |
| 모델이 파일을 어떻게 고치나 | `agent/tools.py` → `_tool_edit_file()` |
| 왜 워크스페이스 밖으로 못 나가나 | `agent/tools.py` → `Workspace.resolve()` |
| 검증 루프 | `agent/verifier.py` + `loop.py` → `_verify_and_repair()` |

### 테스트가 곧 명세입니다

GPU 없이 299개가 다 돕니다. 어떤 함수가 뭘 보장하는지 궁금하면 테스트를 보세요.

| 테스트 | 대상 |
|---|---|
| `test_router.py` | 라우팅 결정 (한국어 포함) |
| `test_classifier.py` | 학습된 분류 + 결과 라벨링 |
| `test_video.py` | 키프레임 선택 정책 |
| `test_tools.py` | 샌드박스 · 파일 조작 · 셸 차단 |
| `test_retrieval.py` | 하이브리드 검색 + 각 단계 폴백 |
| `test_fusion.py` | RRF |
| `test_budget.py` | 시각 토큰 예산 |
| `test_transcript.py` | 세션 기록 + 첨부 리댁션 |
| `test_training.py` | 학습 데이터 수집 + SFT/DPO 내보내기 |
| `test_loop_markers.py` | 히스토리 트리밍 시 캡처 인덱스 |
| `test_web.py` | HTTP 전송 · 업로드 차단 · 승인 왕복 · 두 모드 라우트 |
| `test_settings.py` | 설정 검증 · 전부 아니면 전무 적용 · local.yaml 병합 |
| `test_plan.py` | 파인튜닝 파라미터 검증 · 경고 · 생성된 스크립트 |
| `test_editor_setup.py` | F5가 웹을 띄우는지, 셋업이 web extra를 까는지 |
| `test_bm25.py` / `test_config.py` | 검색 / 설정 |

VS Code 왼쪽 **플라스크 아이콘(Testing 패널)** 에서 개별 실행·디버깅됩니다.

---

## 개발

```bash
pytest -q                    # 299개 테스트 (GPU 불필요)
ruff check src tests         # 린트 (설정은 pyproject.toml의 [tool.ruff])
newal index                  # 저장소 색인만
newal config                 # 병합된 설정 확인
newal config -s router       # 한 섹션만
```

```
src/newal/
├─ config.py          설정 로딩 (YAML 4계층 + 환경변수)
├─ data/default.yaml  기본 설정 (패키지에 동봉)
├─ cli.py             대화형 터미널
├─ web/               브라우저 UI (FastAPI + 순수 HTML/JS, 빌드 없음)
├─ models/            모델 풀 · 라우터 · 분류기 · 임베딩/재정렬
├─ backends/          vLLM/SGLang 클라이언트 · 서버 기동 · transformers 폴백
├─ media/             이미지/동영상 → 콘텐츠 파트, 시각 토큰 예산
├─ memory/            SQLite · BM25 · RRF 융합 · 증분 색인
└─ agent/             도구 · 프롬프트 · 실행 루프 · 검증기
```

`AGENTS.md` / `CLAUDE.md` / `.newal.md`가 있으면 시스템 프롬프트에 자동 반영됩니다.

### 버전 정책

- **4.0** — 브라우저 UI (현재)
- **3.0.2** — LICENSE 추가, `pip install`로 설치하면 기동 못 하던 버그 수정
- **3.0.1** — 히스토리 트리밍이 캡처 인덱스를 어긋나게 하던 버그 수정
- **3.0** — 파인튜닝 데이터 수집 + `newal export`
- **2.1.1** — `ui.transcript_dir`이 선언만 되고 동작하지 않던 버그 수정
- **2.1** — VS Code 전환, `newal config` 추가, 검색 인터페이스 타입 정리
- **2.0** — 학습된 자동 모델 선택
- **1.0** — 이종 모델 풀 + 휴리스틱 라우터
- **0.1** — 단일 모델 + 검증 루프

기능/모델 업데이트는 major, 버그 수정은 minor로 올립니다.

---

## 참고 문헌

- [When Does Combining Language Models Help? A Co-Failure Ceiling](https://arxiv.org/abs/2606.27288) — 앙상블 상한, Self-MoA의 한계
- [Cluster, Route, Escalate: Cost-Aware LLM Serving](https://arxiv.org/pdf/2606.27457) — 캐스케이드 라우팅
- [UCCI: Calibrated Uncertainty for Cost-Optimal Cascade Routing](https://arxiv.org/pdf/2605.18796)
- [When to Think Deeply: Inhibitory Deliberation](https://arxiv.org/pdf/2606.06745) — 적응형 thinking
- [Qwen3 Embedding & Reranker](https://qwenlm.github.io/blog/qwen3-embedding/)
- RouteLLM / RouterDC / MixLLM — 학습된 쿼리 라우팅 계열

## 라이선스

newal은 **MIT**입니다 (`LICENSE`).

의존성은 전부 permissive입니다 (MIT / BSD / Apache-2.0). 자세한 내역은
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md)를 보세요.

한 가지만 알아두시면 됩니다. `video` extra의 opencv-python 휠에는 **FFmpeg이
같이 들어 있고 LGPL**입니다 (GPL 전용 컴포넌트는 빠진 빌드라 GPL이 아닙니다).
newal은 의존성 이름만 적을 뿐 휠을 배포하지 않으므로 이 저장소에 붙는 의무는
없습니다. 다만 `cv2`가 든 컨테이너 이미지나 설치본을 **재배포**하실 거라면
그때는 LGPL 조건을 확인하셔야 합니다.

모델 가중치는 이 라이선스와 무관하며 모델마다 다릅니다. 쓰실 모델의 카드를
직접 확인하세요.

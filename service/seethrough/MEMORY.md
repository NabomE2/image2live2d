# See-through 메모리 병목 분석 및 8GB GPU 실행

대상: Dockerfile이 고정한 See-through `58a1cb11d13f85acec9bbddb8cd4b6487843d4cf`.
RTX 4060 8GB, Windows 11/WSL2, RAM 32GB에서 발생한 900초 초과 문제에 대한 소스 분석입니다.
**실제 RTX 4060의 속도·최대 VRAM/RAM·출력 품질은 미검증입니다.** 이 개발 환경에는 CUDA GPU와 모델 가중치가 없으며, 사용자의 실패 실행에 대한 단계별 로그도 없어 지연의 기여도를 수치로 확정할 수 없습니다.

## 확인한 병목

| 위치 | 현재 소스 동작 | 영향 및 대응 |
| --- | --- | --- |
| `common/utils/inference_utils.py`의 `apply_layerdiff`, `apply_marigold` | 두 파이프라인을 각각 전역 변수에 캐시 | 깊이 모델 로드 때 LayerDiff도 남습니다. 단계별 프로세스를 종료한 뒤 다음 모델을 로드합니다. |
| 같은 파일의 `from_pretrained`와 `.to` | 기본 dtype으로 로드 후 CUDA/BF16으로 변환 | 호스트 메모리 로딩 피크를 줄이도록 BF16 + `low_cpu_mem_usage=True`로 직접 로드합니다. |
| LayerDiff 로드 순서 | UNet/VAE와 두 텍스트 인코더가 동시에 CUDA에 올라간 뒤 프롬프트 캐시 | 텍스트 인코더만 먼저 CUDA에서 캐시 생성 후 해제합니다. 그 다음 기존 코드가 UNet/VAE를 옮깁니다. |
| LayerDiff 본체·머리 추론 | 기본 1280 해상도, 기본 30 steps, 여러 파츠 latent를 함께 처리 | 활성 텐서 피크는 여전히 큽니다. 해상도·파츠·steps·attention·가중치 양자화는 변경하지 않았습니다. 단일 단계 자체가 8GB를 넘을 가능성은 남습니다. |
| 깊이 추론 및 PSD 조립 | 이미지/깊이 numpy 배열과 합성 이미지 보관 | 단계 격리로 GPU 모델과 PSD 조립 메모리가 겹치지 않습니다. PSD 단계도 upstream 모듈을 import하므로 Python/torch의 기본 RAM은 필요합니다. |
| HTTP 서비스 | stdout/stderr 전체 캡처, 요청별 대기 스레드, PSD 전체 읽기 | 로그 스트리밍 + 제한된 오류 tail, 작업 중 HTTP 429, 업로드/결과 파일 스트리밍으로 변경했습니다. |

GPU 91%만으로 병목을 판정할 수 없습니다. VRAM 포화·공유 메모리·RAM 98%는 메모리 압박 및 페이지 이동에 부합하지만, 실제 paging 시간은 Windows/WSL 측정이 필요합니다. 900초만 늘리면 이 문제를 해결하지 못합니다.

## 실행 방식과 보존 범위

`SEE_THROUGH_MEMORY_PROFILE=staged`가 기본입니다. 기존 CLI를 `runpy`로 실행하면서 선택한 함수만 실행합니다. LayerDiff(본체+머리) → 깊이 → PSD를 서로 다른 프로세스에서 실행합니다. 본체와 머리는 **한 번 로드한 동일 모델**을 사용합니다. 매 요청 모델을 다시 로드하는 비용과 세 번의 import 비용은 남지만, 요청 사이 VRAM 잔류를 방지합니다.

기존 seed 42, LayerDiff 1280/30 steps, 깊이 768/모델 기본 steps, 전처리·분리·PSD 경로와 출력 형식을 유지합니다. 기존 빈 머리/누락 PNG 보호 패치도 유지합니다. BF16은 원래 추론 정밀도이며 새 양자화가 아닙니다. 로딩 및 캐시 순서 변경의 수치적 동일성/품질은 실제 모델로 비교해야 합니다. 원본 PNG의 해상도나 알고리즘을 자동으로 낮추지 않습니다.

기본 staged 프로필에서 CPU/group offload는 RAM 98% 상황을 고려해 적용하지 않았습니다. upstream group offload는 전체 CUDA 이동 **이후** 활성화되므로 로딩 피크도 해결하지 못합니다. 커스텀 파이프라인은 `unet.device`와 VAE 내부 메서드를 직접 사용하므로 일반 Diffusers offload hook을 그대로 적용하면 장치 불일치 위험이 있습니다. staged에서는 NF4, VAE tiling, step 감소를 적용하지 않습니다. 선택형 low_vram의 범위는 아래 별도 절을 참고하세요.

`SEE_THROUGH_MEMORY_PROFILE=legacy` + `INFERENCE_MODE=subprocess`는 비교용 기존 단일 CLI 경로입니다. legacy warm은 기존 호환 어댑터이며 이 최적화/강제 deadline의 적용 대상이 아닙니다. Docker는 Uvicorn worker **1개**로 실행하세요. worker/컨테이너를 늘리면 프로세스별 락이 GPU 중복 로딩을 막지 못합니다.

## 로그와 오류

`docker compose logs -f seethrough`에서 `SEETHROUGH_EVENT` JSON을 확인합니다.

- `process_start`, `imports`, `load.<모델 클래스>`, `layerdiff.prompt_cache`, `infer.<파이프라인>`, `layerdiff/depth/psd`: 시작/종료와 `elapsed_seconds`.
- `sample`: 5초마다 RSS/프로세스 peak RSS, Linux MemAvailable/SwapFree, cgroup v2 current/max/events, CUDA allocated/reserved/프로세스 peak 값. 값은 bytes입니다. CUDA 작업 완료 시간을 얻도록 측정 구간 경계에서 synchronize합니다.
- LayerDiff `infer`는 본체, 이어 머리(존재할 때) 순서입니다. CUDA peak는 해당 **프로세스 전체 누적 피크**이며 단계 차분이 아닙니다. `imports`에는 torch import 시간도 포함합니다.
- `/jobs/{id}`는 `telemetry` 최신 이벤트와 `error_code`를 제공합니다. 로그에는 전체 구간이 남으며 API는 로그 전체를 RAM에 보관하지 않습니다.

| 오류 코드 | 근거/의미 |
| --- | --- |
| `cuda_oom` | CUDA OutOfMemoryError 또는 명시적 CUDA OOM 메시지 |
| `host_memory_exhausted` | Python/CPU allocator 메모리 부족 또는 SIGKILL과 cgroup `oom_kill` 증가 |
| `timeout` | 세 단계가 공유하는 전체 wall-clock 예산 초과; 프로세스 그룹 강제 종료 |
| `process_killed` | OOM 근거 없이 신호로 종료됨. OOM으로 단정하지 않습니다. |
| `inference_failed` | 파일/모델/기타 오류. 원래 오류 tail 포함 |

컨테이너 전체 또는 WSL VM이 죽으면 서비스가 오류 응답을 작성할 수 없습니다. 이 경우 Docker의 OOMKilled 상태와 Windows/WSL 로그를 함께 확인하세요. cgroup 기록은 컨테이너 단위여서 다른 프로세스의 OOM과 겹칠 수 있습니다. Linux 메트릭은 Windows 전체 RAM/공유 GPU 메모리가 아니며, cgroup v1 등 제공되지 않는 값은 생략됩니다. PyTorch VRAM 수치는 CUDA context/다른 프로세스 사용량을 포함하지 않습니다.

## 검증 상태와 재현

CPU 자동 테스트는 실제 자식 프로세스에서 로그 스트리밍·bounded tail·시간 초과/자식 종료·오류 분류를 검사하고, 로더 dtype/캐시 순서·공유 deadline·동시 요청 거절·오류 노출을 검사합니다. 실제 PNG → 테스트용 ML 대역 → 실제 PSD 저장까지 단계 어댑터를 실행합니다. **대역 테스트는 실제 See-through 추론 검증이 아닙니다.** 기존 리깅/Live2D 회귀 테스트도 별도로 실행합니다.

RTX 4060에서는 같은 소유 PNG와 캐시된 같은 모델로 legacy/staged 각각 최소 3회 실행하고 다음을 비교하세요. 메모리 압박이 큰 legacy 비교는 다른 작업을 저장한 후 중단 가능한 상태에서 진행하세요.

1. 모델 설치 완료 후 테스트하여 다운로드 시간을 제외합니다.
2. Docker 로그의 모델 로딩/본체/머리/깊이/PSD 시간, RSS/cgroup peak와 CUDA peak를 기록합니다.
3. Windows 작업 관리자에서 공유 GPU 메모리 및 전체 RAM, WSL swap을 함께 관찰합니다.
4. PSD 레이어 이름·개수·크기·알파 및 시각적 품질과 최종 리깅/ZIP을 비교합니다.
5. 단일 추론 단계 OOM이 계속되면 해당 구간 로그를 기준으로 추가 offload를 설계해야 합니다. 현재 변경만으로 8GB 성공이나 속도 향상을 보장하지 않습니다.

모델/SDK 배포와 라이선스는 변경하지 않습니다. `THIRD_PARTY.md`의 조건이 계속 적용됩니다. 공식 Cubism Core 및 VTube Studio 런타임 검증은 여전히 미검증입니다.

## `low_vram`: RTX 4060 8GB를 위한 선택형 프로필

**구현 및 CPU 회귀 검증 완료, 실제 RTX 4060 8GB 적합성·속도·최대 VRAM/RAM·품질은 미검증입니다.**
기본값은 계속 `staged`입니다. `low_vram`은 해상도와 스텝 수를 줄이므로 같은 seed여도
고품질 모드와 같은 결과를 보장하지 않습니다. OOM 시 자동으로 품질을 낮추는 fallback은 없습니다.

| 설정/동작 | `staged` (기본 고품질) | `low_vram` |
| --- | --- | --- |
| LayerDiff 본체·머리 해상도 | 1280 | 768 |
| LayerDiff 스텝 | 30 | 20 |
| 깊이 해상도/스텝 | 768 / 모델 기본값 | 동일 |
| seed / scheduler / BF16 | upstream seed 42 / 기존 scheduler / BF16 | 동일 |
| 단계 격리 / 공유 timeout | LayerDiff → 깊이 → PSD / 900초 | 동일 |
| LayerDiff 모델 배치 | UNet·VAE 동시 CUDA | 인코딩: SD VAE, denoise: UNet, 디코딩: SD VAE+투명도 decoder |
| Attention / FF | upstream 동작 | SDPA / 최대 1024토큰씩 FF 처리 |
| SD VAE | upstream 동작 | slicing + 512픽셀 overlap tiling |

### 설정 및 실행

저장소 루트 `.env`의 기존 항목을 수정하거나 다음 항목을 추가합니다:

```dotenv
SEE_THROUGH_MEMORY_PROFILE=low_vram
SEE_THROUGH_TIMEOUT=900
# Leave client timeout unset to follow independent server stage budgets.
SEE_THROUGH_LAYERDIFF_TIMEOUT=1800
SEE_THROUGH_DEPTH_TIMEOUT=1800
SEE_THROUGH_PSD_TIMEOUT=900
```

WSL 터미널에서 새 어댑터를 포함하도록 이미지를 빌드하고 컨테이너를 재생성합니다.
단순 `docker compose restart`는 변경된 환경변수를 적용하지 않습니다.

```bash
docker compose --profile gpu up -d --build
# 실제 선택된 프로필 확인 (서비스 포트를 외부에 공개하지 않음)
docker compose exec seethrough python -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')))"
docker compose logs -f seethrough
```

`/health`의 `memory_profile`이 `low_vram`인지 확인한 뒤 모델 설치와 PNG 업로드를 진행합니다.
`staged`로 복귀하려면 `.env`를 `SEE_THROUGH_MEMORY_PROFILE=staged`로 바꾸고 같은 `up -d` 명령을
실행합니다. 모델 볼륨을 지울 필요는 없습니다. 일회성 실행은
`SEE_THROUGH_MEMORY_PROFILE=low_vram ./start.sh --gpu`도 가능합니다.

VM에서 직접 실행할 때는 모든 `service/seethrough/*.py`를 같은 디렉터리에 복사한 뒤,
고정 upstream 및 모델이 설치된 환경에서 실행합니다:

```bash
SEE_THROUGH_MEMORY_PROFILE=low_vram SEE_THROUGH_DIR=/opt/see-through \
  uvicorn app:app --host 127.0.0.1 --port 8000
```

서비스 프로세스에 기존 `RESOLUTION`이 설정되어 있으면 그 값이 768보다 우선합니다.
기본 Compose는 `RESOLUTION`을 전달하지 않습니다. 768/20 프로필을 비교하려면 직접 설치의
기존 `RESOLUTION` override를 해제하세요. 스텝은 low_vram에서 20으로 고정하고, 깊이 스텝과
seed는 바꾸지 않습니다. `INFERENCE_MODE=warm`을 지정해도 이 프로필은 단계별 subprocess를 사용합니다.

### 구현 범위와 비용

`low_vram.py`는 고정 upstream CLI의 fullpage·batch=1·guidance=1 경로 전용입니다.
지원하지 않는 인자는 조용히 무시하지 않고 오류로 처리합니다. 기존 빈 머리/누락 PNG 보호,
본체·머리 분리 순서, 파츠 목록, PSD 저장과 리깅/Live2D 내보내기는 그대로 사용합니다.

- 프롬프트를 캐시하고 인코더를 해제한 후에만 UNet을 올립니다. upstream의 초기 VAE CUDA
  이동은 지연하여 첫 호출 전부터 동시 점유를 막습니다. CUDA generator와 seed를 유지합니다.
- 각 전환에서 비활성 모델을 CPU로 옮긴 **다음** 필요한 모델을 CUDA로 옮깁니다.
  UNet 가중치는 단계 사이 CPU RAM에 남으므로 VRAM 감소만큼 RAM·PCIe 전송 비용이 생깁니다.
  이 방법은 호스트 RAM 부족을 해결하지 않습니다.
- SDPA는 PyTorch가 지원하는 커널을 선택합니다. 모든 경우에 Flash Attention을 강제하거나
  사용을 보장하지 않습니다. feed-forward는 토큰별 독립 연산만 청크 처리하며, 프레임 간
  attention을 쪼개거나 13/11개 파츠를 버리지 않습니다.
- SD VAE 인코딩/디코딩에 타일링을 적용합니다. **커스텀 투명도 decoder 자체는 타일링하지
  않습니다.** 한 파츠씩 처리하고, 마지막 latent는 CPU에 둡니다. 타일 경계의 픽셀 차이는
  실제 출력으로 확인해야 합니다.
- 인코딩은 upstream의 `use_offset=False` 경로와 같은 VAE 분포 샘플을 사용하며,
  계산해 놓고 버리던 투명도 offset은 생성하지 않습니다. denoise 임시 텐서는 루프마다
  해제하고, 전체 conditioning은 디코딩 전에 함수 범위를 벗어나도록 분리합니다.
- 깊이 모델은 기존 staged 방식 그대로 별도 프로세스에서 실행합니다. 이 변경의 세부
  UNet/VAE 교대 배치와 FF 청크 최적화는 LayerDiff에 적용됩니다. 깊이 단계 OOM까지 해결했다는
  의미는 아닙니다.

로그의 `layerdiff.low_vram`은 SDPA·청크·타일 설정을, `layerdiff.placement`의 `models`는
현재 CUDA에 배치한 모델을 표시합니다. 기존 5초 샘플 및 `cuda_oom`/`timeout` 분류는 유지합니다.
시간 제한은 자동으로 늘리지 않습니다.

### 재현 가능한 검증

일반 개발 테스트와 별도로 실제 CPU PyTorch/Diffusers 연산 검증을 실행할 수 있습니다.
CUDA Docker 환경에서는 이미 설치된 CUDA PyTorch를 CPU wheel로 교체하지 마세요.
아래 CPU 설치 명령은 GPU 없는 별도 개발 가상환경용입니다.

```bash
pip install -e '.[dev,app]'
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install diffusers==0.37.0
python -m pytest -q -ra tests/test_low_vram.py tests/test_seethrough_memory.py
python -m pytest -q -ra
```

테스트는 단계 배치 순서, 첫 CUDA 이동 지연, staged/legacy 명령 불변, 공유 deadline,
warm 비활성화, 실제 SDPA/GEGLU, 나머지가 있는 FF 청크의 수치 일치, 본체 13/머리 11파츠,
알파 마스크 및 20스텝 처리와 실제 PNG/PSD 어댑터 경로를 확인합니다.
CPU 테스트는 모델 가중치나 CUDA 추론을 실행하지 않습니다. torch/diffusers가 없는 일반
개발 환경에서는 해당 텐서 테스트를 명시적으로 skip합니다. 별도 CI job은 이 의존성을 설치합니다.

RTX 4060에서는 동일 PNG·모델 revision·seed로 `staged`와 `low_vram`을 각각 최소 3회 측정하세요.
다운로드와 첫 로딩을 구분하고, 본체/머리/깊이 시간, CUDA peak, RSS/cgroup peak,
Windows 공유 GPU 메모리·RAM, 생성 PSD의 파츠·경계·알파 및 Live2D ZIP을 비교합니다.
**실측 전에는 “8GB에서 성공” 또는 “몇 배 빨라짐”으로 해석하지 마세요.**


For low_vram timeouts, see [step timing, measured placement and performance analysis](PERFORMANCE.md).

Current staged/low_vram runs use durable checkpoints and independent stage deadlines;
see [stage resume and retention](CHECKPOINTS.md).

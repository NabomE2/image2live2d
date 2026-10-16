# Windows 11에서 PNG → Live2D 모델 ZIP 만들기

이 프로그램은 브라우저에서 캐릭터 PNG를 받아 See-through로 파츠를 분리하고,
메시·리깅·물리를 생성한 다음 Live2D 파일을 ZIP으로 내려줍니다.
현재 공식 Cubism Core 및 Windows VTube Studio에서의 최종 호환성은 **미검증**입니다.
자동 생성 품질은 원본 그림에 따라 달라지며 수동 보정이 필요할 수 있습니다.

## 1. 준비

- Windows 11, WSL2 및 Ubuntu 배포판
- Docker Desktop: Linux 컨테이너, `Use the WSL 2 based engine` 활성화,
  Settings → Resources → WSL Integration에서 사용할 Ubuntu 활성화
- Git: Ubuntu에서 저장소를 받을 때 사용
- PNG 한 장의 자동 분해: WSL 지원 NVIDIA Windows 드라이버와 CUDA GPU 필요
- 인터넷 연결과 디스크 여유 공간: 모델 볼륨에 최소 20GiB, 별도로 Docker 이미지 공간

GPU가 없으면 이미 분리된 PSD 또는 PNG 레이어 ZIP으로 리깅·내보내기를 사용할 수 있습니다.
PNG 자동 분해의 최소 VRAM·소요 시간은 아직 기기별로 검증하지 않았습니다.
CUDA 12.8 기반 컨테이너에 맞는 드라이버를 사용하세요.
WSL에 별도의 Linux 디스플레이 드라이버를 설치하는 방식은 사용하지 마세요.
공식 안내: [Docker GPU 지원](https://docs.docker.com/desktop/features/gpu/),
[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

먼저 [제3자 구성요소와 모델 사용 조건](THIRD_PARTY.md)을 확인하세요.
모델 사용 권한, 원본 그림의 권리, Live2D 관련 이용 조건은 별도입니다.
이 프로젝트의 Apache-2.0 라이선스가 그 권한들을 대신 부여하지 않습니다.

## 2. 실행

Docker Desktop을 실행한 다음 **Ubuntu/WSL 터미널**에서 실행하세요.
저장소는 WSL의 홈 디렉터리에 두는 것을 권장합니다.

```bash
git clone https://github.com/NabomE2/image2live2d.git
cd image2live2d
./start.sh --gpu
```

처음에는 컨테이너 빌드와 Python/CUDA 의존성 다운로드 때문에 오래 걸릴 수 있습니다.
Windows 브라우저에서 `http://localhost:8000`을 직접 여세요.
GPU를 사용하지 않는 PSD/ZIP 작업은 `./start.sh`로 앱을 실행할 수 있습니다.

## 3. 모델 준비와 PNG 업로드

1. 앱 상단에서 GPU 상태를 확인하세요.
2. 모델 카드의 사용 조건을 확인한 뒤 **Download See-through models**를 누르세요.
3. 모델 설치가 끝나면 사용할 권한이 있는 캐릭터 PNG를 선택하거나 끌어 놓으세요.
4. **Run pipeline**을 누르고 분해·메시·리깅 단계가 끝날 때까지 기다리세요.
5. Live2D 파일 생성 및 참조 검사가 통과하면 **VTube Studio용 모델 다운로드** 버튼이 활성화됩니다.

추론에는 시간이 걸립니다. 모델 설치 완료는 추론 성공을 의미하지 않습니다.
분해 또는 Live2D 생성이 실패하면 오류를 확인하세요. 파일이 누락된 ZIP은 제공하지 않습니다.
기존 **Download .inp**는 nijilive용이며 VTube Studio용 파일이 아닙니다.
QA FAIL은 파일 구조 검사와 별개로 모델의 품질 문제가 발견됐다는 뜻입니다.

## 4. Windows에서 다운로드 찾기와 VTube Studio로 가져오기

1. ZIP 다운로드 후 브라우저에서 **Ctrl+J** → **폴더에서 표시**를 누르세요.
   기본 설정에서는 Windows 사용자의 `다운로드` 폴더에 저장됩니다.
   파일 이름은 `character_vtube.zip`과 같은 형태이며 Windows에서 안전한 이름으로 정리됩니다.
2. ZIP을 우클릭 → **모두 압축 풀기**를 선택하세요.
3. 압축을 푼 모델 폴더 전체를 Windows의 VTube Studio `Live2DModels` 폴더로 복사하세요.
   Steam의 설치 파일 탐색 기능으로 VTube Studio 설치 위치를 찾을 수 있습니다.
4. `.model3.json`, `.moc3`, `textures` 폴더와 나머지 JSON을 함께 유지하세요.
5. VTube Studio에서 모델을 선택하고 추적 입력·파라미터 범위·물리를 설정하세요.

ZIP 내부에는 한 모델 폴더와 사용 안내가 들어갑니다.
공식 Core 로드·렌더링, VTube Studio 가져오기, 얼굴 추적 및 물리 품질은 **미검증**입니다.
파일 참조 검사를 통과했다는 표시가 이 기능들의 인증을 뜻하지 않습니다.
VTube Studio 자체를 WSL/Docker 안에서 실행할 필요는 없습니다.

## 5. 문제 해결과 종료

```bash
docker compose logs --tail=100 app
docker compose --profile gpu logs --tail=100 seethrough
docker compose --profile gpu down
```

- GPU가 없다고 나오면 WSL의 `nvidia-smi` 및 Docker Desktop GPU 설정부터 확인하세요.
- GPU 메모리 부족은 로그에서 확인하세요. 현재 모든 NVIDIA GPU의 성공을 보장하지 않습니다.
- 다운로드 실패 시 연결·디스크 공간·모델 접근 권한을 확인하세요. 접근 제한 우회는 지원하지 않습니다.
- 8000 포트 충돌 시 `PORT=8001 ./start.sh --gpu` 후 Windows에서 8001 포트를 여세요.
- 업로드와 생성 결과는 임시 파일입니다. 컨테이너를 재생성하기 전에 ZIP을 다운로드하세요.
- 일반 `down`은 모델 볼륨을 유지합니다. `down -v`는 다운로드한 모델도 삭제하므로 주의하세요.

## 개발 및 검증

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,app]'
python -m pytest -q -ra
```

일반 테스트의 PNG 입력은 프로젝트가 생성한 유효한 PNG입니다. HTTP 테스트용 분해 서비스가
실제 PSD를 반환하며, 실제 클라이언트·PSD 파서·리깅·내보내기·ZIP 다운로드를 실행합니다.
**이 테스트는 GPU 추론을 실행하지 않습니다.**

본인 소유 PNG와 준비된 실제 See-through 서비스를 이용한 전체 테스트는 별도로 실행합니다.
테스트 프로세스에서 해당 서비스에 접근할 수 있어야 합니다.

```bash
I2L_TEST_PNG=/path/to/your-character.png \
I2L_TEST_DECOMPOSE_URL=http://127.0.0.1:8000 \
python -m pytest tests/test_live2d_download.py::test_real_png_gpu_pipeline -q -ra
```

위 URL은 **분해 서비스** 주소이며 웹 앱 주소가 아닙니다.
Compose의 기본 분해 서비스는 호스트에 포트를 공개하지 않습니다.
필요하면 로컬 테스트용 Compose override에서만 별도의 loopback 포트를 설정하세요.
인증 서비스는 `IMAGE2LIVE2D_DECOMPOSE_TOKEN`을 환경에 안전하게 설정하세요.
PNG와 서비스 설정이 없으면 해당 테스트는 명시적으로 건너뜁니다.
공식 Core 및 VTube Studio 테스트는 별도로 수행해야 합니다.

CLI와 Python API도 같은 내보내기를 사용합니다.

```bash
python -m image2live2d --image character.png --decompose-url http://YOUR_SERVICE:8000 \
  -o out/character.inp --live2d out/character_live2d
```

```python
from image2live2d import convert_image

result = convert_image("character.png", "out", service_url="http://YOUR_SERVICE:8000")
print(result.live2d_path, result.live2d_zip_path)
```

CLI ZIP은 지정한 Live2D 디렉터리 옆에 생성되며, API에서는 `live2d_zip_path`로 확인할 수 있습니다.

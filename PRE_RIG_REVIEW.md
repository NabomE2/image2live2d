# Pre-Rigging Studio: 검수 게이트 MVP

PR #8 (`130f639`, `feat/mouth-detail-editor`) 위에 추가한 기능입니다. PR #8의
완성 모델용 EditorStore를 사용하지 않고 별도 ReviewStore를 사용합니다.
원자적 JSON 저장, fsync, 프로젝트 잠금 등의 저장 primitive만 공유합니다.

## 사용

1. 업데이트한 앱을 실행합니다. Docker는 `docker compose up -d --build app`.
2. localhost:8000 업로드 화면에서 **수동 검수 후 리깅**을 선택합니다.
3. PNG 분해가 완료되면 `/review?project=...` 검수 화면으로 이동합니다.
4. 파츠 목록에서 단독 표시 또는 전체 합성을 선택하고 격자/RGB/알파로 확인합니다.
5. 준비 정책을 확인하고 **승인 후 리깅 시작**을 누릅니다.
6. CPU 처리가 끝나면 완전한 VTube Studio ZIP을 다운로드합니다.

자동 모드가 기본값이며 기존 연속 생성 흐름을 유지합니다.
PSD 및 파츠 PNG ZIP 입력도 검수할 수 있습니다. ZIP 입력에는 원본 PSD가 없으므로
새 PSD를 만들지 않습니다. PNG 입력에는 분해 PSD가 있어야 검수 스냅샷을 게시합니다.
원본 이미지와 정규화 파츠는 캔버스가 다를 수 있어 나란히 비교하며 픽셀 차이 점수는
제공하지 않습니다. 원본 ZIP 등 이미지로 열 수 없는 입력은 원본 미리보기 대신 안내합니다.

서버/Docker 재시작 후 첫 화면의 **검수 프로젝트** 링크 또는 `/review`에서 다시 엽니다.
임시 생성 job이 만료되어도 이 목록에서 승인 및 다운로드할 수 있습니다.

## 중단과 재개

`server._run_pipeline`은 Decompose 완료 직후 수동 모드라면 스냅샷을 저장하고
`review_required`로 종료합니다. `prepare_meshes`, landmarks, rig, physics, motion,
export는 실행하지 않습니다. 관리형 GPU 연결은 기존 finally에서 해제합니다.

승인 요청은 스냅샷의 파일 해시, 요청 content_hash 및 준비 정책을 검사합니다.
승인된 원본 파츠를 새 attempt 디렉터리에 복사하고 같은 CPU 파이프라인의
Build meshes부터 실행합니다. 분해 단계 및 GPU manager는 건너뜁니다.
완성된 ZIP과 IRR을 검증·동기화한 후 last_good 포인터를 원자적으로 게시합니다.

승인 후 **기존 자동 준비 정책**을 유지합니다: 누락 파츠/구강 합성, 팔다리 분할,
역할·좌우 보정, 얼굴·다리·장신구 순서 보정. 이 MVP는 파츠를 수정하거나 잠그는
편집기가 아닙니다. 원본 스냅샷과 실제 리깅 입력은 별개이며, 보정은 작업 복사본에만
적용됩니다. 준비 정책은 버전과 설치된 Python 소스 해시를 함께 고정합니다.
앱 코드가 바뀐 후 기존 프로젝트 승인은 안전하게 거절합니다. 이 경우 원래 앱 버전으로
열어야 합니다. 정책 마이그레이션 및 재승인 UI는 후속 작업입니다.

## 저장 구조와 보존

기본: `IMAGE2LIVE2D_JOB_DIR/review-projects` (Docker 기본 `/jobs/review-projects`).
`IMAGE2LIVE2D_REVIEW_DIR`로 별도 영속 볼륨을 지정할 수 있습니다.

```
review-projects/<project-id>/
  project.json             # 상태, 해시, 승인 정책, last_good
  original/source          # 업로드 바이트 원본
  original/psd/            # 원본 PSD (존재하는 입력)
  original/parts/          # 정규화 파츠 원본
  original/snapshot.json   # 캔버스, 파츠 역할/순서/좌표, 정책, 추론 receipt
  attempts/<attempt-id>/   # CPU 처리 복사본, normalized, rig.json, live2d, model.zip
```

PR #8의 `editor-projects` 및 임시 job 디렉터리와 분리됩니다. 검수 프로젝트는
7일 정리에 포함하지 않습니다. 이 MVP는 원본 revision 한 개와 승인/빌드 시도를
보존합니다. 원화 편집 revision UI나 자동 장기 보관 정리는 아직 없습니다.
디스크 사용량은 늘어나므로 앱 종료 후 백업 및 불필요한 프로젝트를 수동 정리하십시오.
`docker compose down -v`는 프로젝트가 든 볼륨도 삭제합니다.

## 승인·실패·동시성

- 같은 Origin의 JSON 요청만 승인합니다. `approved: true`와 정확한 content_hash 필요.
- 파일 잠금으로 프로젝트 상태 변경을 직렬화하고 별도 build 잠금으로 프로세스 간
  중복 실행을 막습니다. 완료 프로젝트의 재승인은 기존 결과를 반환합니다.
- 프로세스가 중단되어 build 잠금이 해제되면 `interrupted`로 표시합니다.
  명시적 승인 재시도로 새 attempt를 만듭니다. 자동으로 재실행하지 않습니다.
- 실패한 attempt는 원본 및 기존 last_good을 덮어쓰지 않습니다.
- 출력 게시 전 중단된 파일은 다운로드 대상으로 등록하지 않습니다.
- 현재 POSIX 잠금 기반이며 Docker/WSL/Linux/macOS를 대상으로 합니다.
- 분해 실패/Depth 재시도는 기존 PR #6 경로를 사용하며 manual 모드를 유지합니다.
  검수 프로젝트가 생성된 뒤에는 GPU resume 대신 검수 승인 경로를 사용합니다.

## API

- `POST /api/jobs?name=...&mode=auto|manual`: 기본 auto.
- `GET /api/reviews`: 저장된 검수 프로젝트 목록.
- `GET /api/reviews/{id}`: 상태·파츠·해시·정책·승인·마지막 성공 결과.
- `GET /api/reviews/{id}/image?part=composite|source|<part-id>&view=rgba|rgb|alpha`.
- `POST /api/reviews/{id}/approve`: `{content_hash, approved: true}`.
- `GET /api/reviews/{id}/download`: 마지막 정상 모델 ZIP.

## 검증 및 범위

회귀 테스트는 GPU 경계만 대체하고 실제 CPU 메시·리깅·export를 실행합니다.
승인 전 메시 미실행, 자동/수동 IRR 일치, 원본 해시, 별도 프로세스 재열기,
승인 직후 프로세스 종료, 동시 승인, 게시 실패, HTTP 및 Chromium UI를 검사합니다.
전체 기존 테스트로 PR #5~#8 회귀를 확인합니다.

실제 RTX 4060 추론, Docker 재시작 실측, 사용자 모델 품질 및 Cubism/VTube Studio
움직임은 이 CPU 테스트의 검증 범위가 아닙니다. UI 이미지는 정적 검수용입니다.

브러시·파츠 추가/교체/배치·인페인팅·PSD writer·자연어·접합부 보정은 없습니다.
검수 완료 모델에서 PR #8 편집 프로젝트로 직접 전환하는 UI도 이 MVP에는 없습니다.
기존 자동 모드의 PR #8 편집 경로는 유지합니다.
다음 파츠 편집 단계에서는 ArtRevision, 안정된 파츠 ID, 문서/정책 해시, 변경 시
승인 무효화, 자동 준비 보정과 사용자 확정 구조의 충돌 정책을 먼저 추가해야 합니다.

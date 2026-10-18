# 웹 입 알파 디테일 편집기

생성 완료된 모델의 입 알파만 수정하는 CPU 편집기입니다. RGB, 메시, UV,
키폼과 `.moc3`는 보존하며 LayerDiff/Depth를 호출하지 않습니다.

## 사용 방법

1. 업데이트한 앱을 실행합니다(Docker: `docker compose up -d --build app`).
2. localhost:8000에서 모델 생성을 완료하고 **디테일 편집**을 누릅니다.
   이때 원본과 생성 결과를 독립된 편집 프로젝트로 복사합니다.
3. 입 파츠를 선택하고 오른쪽 초안에 지우기/원본 복원 브러시를 사용합니다.
   크기는 원본 파츠 픽셀 단위이고 경도는 가장자리 감쇠를 조절합니다.
   포인터를 놓으면 서버에 저장됩니다. 휠 확대, 이동 도구 또는 Shift로 이동합니다.
4. 왼쪽 원본과 오른쪽 수정본을 비교합니다. 격자/흰색/검정 배경과 알파 보기로
   경계를 확인합니다. 이것은 파츠 이미지 비교이며 전체 모델 움직임 미리보기가 아닙니다.
5. **적용**과 확인을 거치면 새 불변 revision이 생성됩니다. ZIP을 다운로드하여
   VTube Studio에서 확인합니다. 이전 버전 선택 후 복원하면 그 상태의 새 버전이 생깁니다.
6. 서버 재시작 뒤 첫 화면의 저장된 편집 프로젝트 링크(`/editor`)로 다시 엽니다.
   저장된 초안도 목록에서 다시 열 수 있습니다.

입 이외의 파츠는 조회만 가능합니다. 복원 브러시는 원본 알파를 초과하지 않으며,
원본 RGB를 바꾸지 않습니다. 한 초안은 한 입 파츠를 편집합니다.
이전 버전은 조회/다운로드/복원 가능하고 새 편집은 현재 버전에서 시작합니다.

## 보존 및 재시작

기본 저장 경로는 `IMAGE2LIVE2D_JOB_DIR` 아래 `editor-projects`입니다.
`IMAGE2LIVE2D_EDITOR_DIR`로 독립된 영속 디렉터리를 지정할 수 있습니다.
Docker 기본 `/jobs` 볼륨에 저장되며 임시 작업의 7일 정리와 분리됩니다.
프로젝트에는 입력 원본 바이트 및 파일명, 합성/메시 준비 전 정규화 PNG,
최종 파츠, 메시/키폼을 포함한 IRR, 모델 번들, atlas 배치/UV 정보 및 원본 해시가 저장됩니다.
원본과 revision은 API로 덮어쓰지 않습니다.

편집 프로젝트로 전환하기 전의 생성 작업은 메모리의 rig가 필요합니다.
따라서 생성 완료 후 서버를 종료하기 전에 프로젝트를 만들어야 합니다.
이전 앱 버전에서 생성하여 정규화 스냅샷이 없는 작업은 안전하게 거절합니다.
이미 만든 프로젝트는 원래 작업이 정리되거나 서버가 재시작되어도 열 수 있습니다.

승인된 프로젝트는 자동 삭제하지 않습니다. revision마다 완전한 복사본을 저장하므로
디스크 사용량이 증가합니다. 초안의 이전 스트로크 파일과 중단된 임시 export도
자동 정리하지 않습니다. 필요하면 앱을 정지하고 백업 후 불필요한 프로젝트를
수동 정리하십시오. Docker `down -v`는 프로젝트를 포함한 볼륨을 삭제합니다.

## 안전한 텍스처 재출력

프로젝트 생성 시 현재 exporter로 재구성한 atlas 픽셀과 `.moc3` 바이트가
완료된 번들과 동일한지 검사합니다. atlas 배치가 겹치거나 지원되지 않는 경우,
원본 데이터가 불완전하거나 일치하지 않는 경우에는 성공 처리하지 않습니다.
이 MVP는 현재 앱의 단일 atlas export만 지원하며 외부 ZIP 역편집은 지원하지 않습니다.

편집 승인 시 원본 해시, RGB 불변, 원본 이하 알파를 검사합니다.
기존 atlas의 입 tile 알파만 갱신하며 RGB와 다른 tile은 유지합니다.
축소된 tile은 기존 배치 크기로 리샘플링하고 원본 atlas 알파 이하로 제한합니다.
`.moc3` 및 atlas 이외 번들 파일의 해시 동일성을 확인한 뒤 번들을 검증합니다.
메시 경계 밖 픽셀을 새로 만드는 편집은 지원하지 않습니다.

프로세스 파일 잠금, 현재 revision과 초안 sequence 비교로 충돌을 거절합니다(HTTP 409).
다른 창이 먼저 저장했다면 새로고침하고 최신 버전에서 편집해야 합니다.
파일을 임시 위치에 쓰고 fsync/rename한 뒤 프로젝트 manifest를 마지막에 게시합니다.
실패한 export는 기존 정상 버전을 교체하지 않습니다. 복원도 새 revision입니다.
파일 잠금은 POSIX 기반이므로 Docker/WSL/Linux/macOS에서 실행하십시오.

## API

변경 요청은 앱과 같은 Origin의 JSON POST입니다.

| 경로 | 동작 |
|---|---|
| POST `/api/jobs/{job}/editor-project` | 완료 작업을 영속 프로젝트로 복사 |
| GET `/api/editor/projects` | 프로젝트 목록 |
| GET `/api/editor/projects/{id}` | 파츠, revision, 미적용 초안 |
| POST `/{id}/drafts` | `base`, `part_id`로 초안 생성 |
| POST `/{id}/strokes` | `draft`, `sequence`, `strokes` 저장 |
| POST `/{id}/apply` | `draft`, `sequence`, `approved: true`로 게시 |
| POST `/{id}/restore` | `target`, `base`, `approved: true`로 복원 |
| GET `/{id}/asset` | `part`, `revision` 또는 `draft` PNG |
| GET `/{id}/download` | `revision`의 완전한 VTube Studio ZIP |

축약된 경로의 접두사는 `/api/editor/projects`입니다. 각 dab은
`x`, `y`, `radius`, `hardness`, `mode`(`erase`/`restore`)를 사용합니다.
인증을 제공하는 공개 서비스가 아니라 기존 localhost 개인용 앱을 대상으로 합니다.

## 검증과 한계

`tests/test_detail_editor.py`는 재시작 복구(별도 Python 프로세스), 원본 해시,
브러시 범위 밖 픽셀, 다른 atlas tile, moc/UV/키폼 보존, 축소 atlas 복원,
충돌, 중단된 export, 다운로드, 정리 정책 및 GPU 경로 미호출을 검사합니다.
선택적 Playwright+Chromium 테스트는 실제 웹 브러시, 초안 재열기, 승인,
ZIP 다운로드와 버전 복원을 검사합니다. Playwright가 없으면 해당 테스트는 건너뜁니다.

이 검증은 합성 fixture 기반입니다. 실제 사용자 모델의 입 품질 개선,
Cubism Core/VTube Studio의 시각적 움직임, RTX 4060 실행은 별도 검증이 필요합니다.
알파를 지우면 얼굴 베이스 또는 합성 구강이 드러날 수 있습니다. 구강/치아/혀의
생성 방식과 리깅은 이번 변경에서 수정하지 않았습니다.
자연어 편집, 목 추종, 인페인팅, 자동 입선 추출도 포함하지 않습니다.

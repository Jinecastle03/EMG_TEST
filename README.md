# EMG Pose × Effort Monitor (MindRove Armband 2)

MindRove Armband 2의 **RAW EMG 8ch(500 Hz) + IMU 6축(~50 Hz)** 으로 손동작(Pose)과 상대적 힘 수준(Effort Low/High)을
실시간 추정하는 Windows 데스크톱 연구용 프로그램입니다. (Python · PySide6 · pyqtgraph · scikit-learn · PyTorch, Unity 불필요)

> Effort는 지시된 **상대적 근수축 수준(Low/High)** 이며 kgf/N 단위의 힘이 아닙니다.

## 실행

```bat
run_app.bat                       :: GUI (= .venv\Scripts\python -m emgmonitor.app)
train.bat --windows 200 300 500   :: 명령행 학습/비교 (GUI의 Train 버튼과 동일)
.venv\Scripts\python tests\test_system.py          :: 장비 없이 하는 시스템 테스트 (약 5분)
.venv\Scripts\python tests\check_legacy_parity.py  :: 기존 파이프라인과의 수치 일치 검사
.venv\Scripts\python -m emgmonitor.legacy          :: 기존 세션 import (이미 실행됨, 재실행해도 덮어쓰지 않음)
```

새 PC 설치: Python 3.14 → `python -m venv --system-site-packages .venv` → `.venv\Scripts\pip install -r requirements.txt`
(torch는 시스템 Python에 설치된 것을 사용합니다.)

## 사용 순서 (GUI 하단 버튼)

1. **Connect** – `MindRove Armband 2 (Wi-Fi)` 선택. 암밴드 전원 ON → PC Wi-Fi를 암밴드 네트워크에 연결 → 방화벽에서 `python.exe` 허용.
   (`SDK Synthetic board`는 장비 없이 SDK 경로만 시험하는 250 Hz 가짜 신호이며 학습에 쓰이지 않습니다.)
2. 상단 **Participant** 입력 → **Calibrate** (약 20초): ① **가만히** 5초 → ② **손목 굽히기 최대** 3초 → (선택) **손목 펴기 최대** 3초.
   손목은 실제로 움직이지 않고 다른 손으로 막은 채 버티기만 합니다. 장치 점검(EMG/IMU 속도, FLAT/NOISY 채널)은 ①의 5초 데이터로 자동 계산.
   결과: 채널별 rest RMS / max RMS → `data/calibrations/<참가자>/Cal_*.json` (원신호는 kind=calibration 세션).
   용도는 **Normalized 파형 보기**와 **상대 활성도**(0 % = 휴식, 100 % = 손목 최대)뿐이며 **모델 입력에는 쓰이지 않습니다**
   (모델은 사람별로 학습되고 자기 학습 데이터로 정규화). 참가자 ID를 입력하면 최신 캘리브레이션이 자동 로드됩니다.
3. **Collect Data** – 동작 선택(기본 5개 + **Add new pose…** 로 새 동작 추가), 조건당 10회, Rest trial 10회, 준비 2 s / 유지 3 s / 휴식 4 s,
   블록 랜덤화. 큰 글씨 큐(한글 이름 + 구분 포인트 + 물체 + 강도 안내) + 상대 활성도 막대(이번 세션의 Low/High 중앙값 표시).
   trial마다 품질 경고(패킷 손실, 평탄 채널, Low가 이번 세션 High만큼 강함, High가 Low만큼 약함, Rest 중 움직임, gyro 포화). **Redo last trial / Reject selected → re-collect** 로 재수집.
4. **Train** – 세션·모델·윈도(200/300/500 ms)·평가 분할(trial/session/participant) 선택 → 별도 프로세스에서 학습 → 비교표 + 추천 모델 → *Use recommended model*.
5. 우측 **Current model** 선택 → **Live Inference** → POSE / EFFORT 큰 글씨, 신뢰도, 상대 EMG 활성도, 추론/지연 시간.
6. **Replay** – 저장된 세션을 실시간 재생(주황색 `REPLAY MODE` 배너, 정답 TRUTH 표시). **Save (Record)** – 원신호 + 실시간 예측을 세션으로 저장.

새 동작은 코드 수정 없이 추가됩니다: Add new pose → 수집 → Train (모델은 선택한 세션에 존재하는 라벨을 그대로 학습).

## 동작 정의 (데이터 라벨은 영문, 화면은 한글)

| 라벨 | 화면 이름 | 동작 | 구분 포인트 | 물체 |
|---|---|---|---|---|
| Rest | 휴식 | 동작 자세 그대로 힘 빼기 | 아무 힘도 안 줌 | – |
| Power Grip | 주먹 쥐기 | 손바닥 전체 + 네 손가락으로 감싸 쥐기 | 손바닥이 물체에 닿음, 다섯 손가락 사용 | 물병 |
| Precision Pinch | 손끝 집기 (OK 모양) | 엄지 끝 ↔ 검지 끝으로 집기 | 손가락 **끝**끼리, 손바닥 안 닿음 | 동전 |
| Key Pinch | 옆으로 집기 (열쇠 잡기) | 엄지 지문면으로 구부린 검지 **옆면**에 눌러 집기 | 손끝이 아니라 검지 옆면 | 카드/열쇠 |
| Index Press | 검지로 누르기 | 검지만 펴서 책상 누르기 (초인종) | 검지 하나만 힘, 집지 않음 | 책상 |
| Twist CW | 뚜껑 돌리기 (시계 방향) | 위에서 볼 때 시계 방향으로 돌리기 | 전완 회전, 속도는 항상 1회/초 | 병 |

Low = 최대 힘의 약 20–30 %, High = 무리하지 않는 약 60–70 % (속도가 아니라 힘으로 구분).
주의: 두 집기(Precision / Key Pinch)는 엄지 쪽 손바닥 근육(손 안의 근육)의 기여가 커서 전완 암밴드로는 구분이 어려울 수 있습니다.
수집 후 혼동행렬에서 두 클래스가 섞이면 하나로 합치거나 다른 동작으로 바꾸는 것을 권장합니다.

## 모델이 읽는 데이터와 속도

모든 모델은 **50 ms마다(20 Hz)** 가장 최근 윈도 하나를 읽어 예측합니다. GUI 우측 *Model* 칸에 현재 모델의 입력이 그대로 표시됩니다.

| 윈도 | EMG (8ch, 500 Hz, 필터 후) | IMU (acc 3 + gyro 3, 50 Hz) | 연속 예측 겹침 |
|---|---|---|---|
| 200 ms | 100 샘플/ch → (8, 100) | 10 샘플/축 → (6, 10) | 150 ms |
| 300 ms | 150 → (8, 150) | 15 → (6, 15) | 250 ms |
| 500 ms | 250 → (8, 250) | 25 → (6, 25) | 450 ms |

| 모델 | Pose 입력 | Effort 입력 |
|---|---|---|
| Feature RF/LR (기존 포함) | 위 구간에서 계산한 특징 45개: EMG RMS·MAV·WL·STD × 8ch (32) + IMU 통계 13 | EMG 특징 32개 (Pose별 LR) |
| RAW EMG CNN | EMG 시계열 (8, T) | EMG 시계열 (8, T) |
| IMU-only CNN | IMU 시계열 (6, T') | IMU 시계열 (6, T') |
| Two-Stream CNN | EMG (8, T) + IMU (6, T') | 기본 EMG만 (IMU가 속도만 학습하면 자동 제외) |
| Multi-task CNN | EMG + IMU (공유 인코더) | 같은 인코더의 Effort head |

캘리브레이션 값은 어떤 모델의 입력에도 들어가지 않습니다. 처리 지연(새 샘플 수신 → 예측 완료)은 RF 약 20 ms, CNN 약 2 ms (Replay 1× 실측).

## 디렉터리 구조

```
EMGTEST/
  run_app.bat, train.bat, requirements.txt
  emgmonitor/
    config.py        경로, 필터/윈도 설정 (기존 프로젝트와 동일)
    poses.py         동작 레지스트리 (config/poses.json 에 사용자 동작 추가)
    dsp.py           causal 필터, package_num 샘플 클록, IMU 변화 감지, 타임스탬프 정렬 윈도, 기존 특징
    acquisition.py   MindRove SDK(Wi-Fi/Synthetic) 수집 스레드, Replay 소스
    engine.py        실시간 처리 스레드: 필터→링버퍼→녹화→활성도→윈도→모델→평활화 (GUI와 분리)
    session.py       RAW CSV 세션 writer(전용 스레드)/loader
    calibration.py   캘리브레이션 (Rest + 손목 Max)
    dataset.py       라벨 구간 → 윈도, trial/session/participant 분할 (윈도 생성 전 분할)
    deep.py          1D-CNN (EMG / IMU / Two-Stream / Multi-task)
    predictors.py    실시간 예측기 공통 인터페이스 (기존 RF/LR, 새 Feature 모델, CNN)
    trainer.py       학습·CV·평가·비교표·추천·리포트
    legacy.py        기존 세션/캘리브레이션 import (읽기 전용)
    gui/             main_window, calibration_wizard, collection, training, theme
  data/
    sessions/<참가자>/<세션>/  emg.csv imu.csv trials.csv events.csv metadata.json [predictions.csv calibration.json]
    calibrations/<참가자>/*.json
    logs/            mindrove_sdk.log, 학습 로그
  models/
    legacy_20261103/ 기존 RF/LR 모델 복사본
    runs/<run_id>/   report.md, results.json, comparison.csv, w<ms>/<모델>/{model.joblib|model.pt, meta.json, confusion_test.png}
  tests/             test_system.py, check_legacy_parity.py
```

## 데이터 형식 (원본 보존)

- `emg.csv`: `sample_index, package_num, device_ts, host_ts, trial_id, emg_ch1_uv … emg_ch8_uv` — SDK가 준 RAW 값(필터 없음, DC 포함)
- `imu.csv`: `imu_index, sample_index, package_num, device_ts, host_ts, trial_id, acc_x_g … gyro_z_dps`
- `trials.csv`: trial별 `pose, effort, repetition, status(completed/rejected/aborted), prep_start_idx, hold_start_idx, hold_end_idx, rest_end_idx`, 품질 경고
- `metadata.json`: 참가자, 세션, kind(training/calibration/free), 데이터 출처(DEVICE/SYNTHETIC/REPLAY/IMPORTED_LEGACY_DEVICE),
  SDK 행 레이아웃, 샘플링 속도, 필터, 프로토콜, 캘리브레이션 스냅샷, 동작 설명
- 필터·윈도·특징은 학습 시점에만 계산하며 원본 파일은 다시 쓰지 않습니다.

## 신호 처리 / 시간 정렬

- `device_ts`는 패킷 단위 PC 시각이라 샘플 시각으로 쓸 수 없음 → `package_num`(샘플당 +1, 손실 시 +k)을 펼친 **샘플 클록**을 공통 시간축으로 사용.
  IMU 샘플은 그것이 들어온 EMG 행의 클록을 받음(SDK가 IMU를 같은 500 Hz 행 스트림에 sample-and-hold로 넣기 때문, 값 변화 감지로 ~50 Hz 추출).
- 윈도 (t_end − W, t_end]: EMG/IMU를 **각자의 클록 값으로** 선택(배열 길이로 맞추지 않음). EMG 누락 샘플은 선형 보간(커버리지 < 80 %면 무효),
  IMU는 50 Hz 격자에 인과적 sample-hold (200 ms → EMG 8×100, IMU 6×10).
- EMG 필터: 20 Hz 4차 high-pass + 60 Hz notch(고조파 포함), **인과(causal)** IIR, 첫 샘플로 상태 초기화 — 기존 Unity/Python과 동일(오차 0.0 µV 확인).
  학습과 실시간이 **같은 함수**(`dsp.extract_window`, `predictors.*.predict`)를 사용하고, 학습 시 매 모델마다 재추출 윈도로 parity를 검사합니다.

## 모델

| 모델 | 입력 | 구조 |
|---|---|---|
| A Feature-based (Baseline) | RMS/MAV/WL/STD ×8 + IMU 통계 13 | log1p → StandardScaler → RF(300 trees, Pose) + Pose별 Logistic Regression(Effort) — 기존 코드 재사용 |
| B RAW EMG 1D-CNN | 필터된 EMG 8×T | Conv(32,k7)-Pool-Conv(64,k5)-Pool-Conv(64,k5) → [GAP‖GMP] → FC64 (≈41 k params) |
| C IMU-only 1D-CNN | IMU 6×T' | Conv(16,k3)-Conv(32,k3) → [GAP‖GMP] → FC64 (≈6 k) |
| D Two-Stream CNN | EMG 8×T + IMU 6×T' | B 인코더 ‖ C 인코더 → concat → FC64 → Pose head (≈47 k) |
| D-MT Multi-task | 동일 | 공유 인코더 + Pose head + Effort head (Rest는 effort loss 제외) |

- Effort head: **Pose별 Low/High 출력**(pose-conditioned). 학습은 실제 Pose의 출력만, 실시간은 예측된 Pose의 출력 사용 = Hierarchical.
- Effort 입력은 EMG-only 기본. EMG+IMU는 CV에서 (1) BA가 0.01 넘게 좋고 (2) **속도 혼입(같은 effort 안에서 회전속도↔P(High) Spearman ρ)이
  0.10 이상 증가하지 않을 때만** 채택 → Twist Low/High를 회전 속도로 구분하는 문제를 자동 배제.
- 정규화 통계는 학습 fold의 윈도에서만 계산. AdamW + weight decay, dropout, 채널 gain/noise 증강, 학습 trial 20 %로 early stopping.
- 모델 선택은 dev CV만 사용(테스트는 1회 평가). Deep 모델이 Baseline보다 CV에서 0.01 넘게 좋지 않으면 Baseline 추천.

## 평가 (기존 세션 S01, 4 class, trial-independent, test = 라벨당 마지막 2 trial = 14 trial)

`models/runs/legacy_S01_cv/report.md` 전문 참조. 단일 참가자·단일 세션 → **Session/Participant-independent 평가는 불가능**(프로그램이 명시함).

| Model | Window | Pose Macro-F1 test (CV) | Effort BA, true pose test (CV) | Effort BA cascade | CPU latency |
|---|---|---|---|---|---|
| Feature RF/LR (baseline) | 200 ms | 0.929 (0.859) | 0.740 (0.805) | 0.730 | 13.7 ms |
| RAW EMG CNN | 200 ms | 0.877 (0.770) | 0.777 (0.794) | 0.759 | 0.6 ms |
| IMU-only CNN | 200 ms | 0.878 (0.775) | 0.672 (0.547) | 0.674 | 0.3 ms |
| EMG+IMU Two-Stream | 200 ms | 0.951 (0.837) | 0.772 (0.790) | 0.766 | 0.9 ms |
| Feature RF/LR (baseline) | 500 ms | 0.969 (0.867) | 0.813 (0.851) | 0.812 | 15.4 ms |
| EMG+IMU Two-Stream | 500 ms | 0.981 (0.872) | 0.807 (0.858) | 0.809 | 1.2 ms |

CNN 수치는 GPU 학습의 비결정성 때문에 실행마다 ±0.01 정도 달라집니다.

자동 추천: **Feature RF/LR @ 500 ms** (CV 기준; Two-Stream은 Baseline보다 0.01 넘게 좋지 않음).

## 실제 장치 검증

**실제 MindRove 장비로는 아직 테스트하지 않았습니다.** 확인된 것은 SDK Synthetic board, 기존 녹화 데이터 Replay, offscreen GUI 자동 테스트입니다.
장비 연결 시 확인할 항목: 상단 EMG ≈500 Hz / IMU ≈50 Hz, `lost` 증가 없음, 8채널 파형/FLAT 경고 없음, 캘리브레이션 Device check 통과,
Save로 30초 녹화 후 Replay에서 같은 파형, Live Inference 시 20 pred/s, 지연 표시.

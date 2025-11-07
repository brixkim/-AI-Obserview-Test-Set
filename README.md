# -AI-Obserview-Test-Set
AI Obserview 테스트 환경

## 개요
이 프로젝트는 `YOLOv8n` 이후 최신 버전의 `YOLO`모델을 테스트하기 위한 환경입니다.<br/>
![ver.nano bench](/assets/images/yolo_bench.png)<br/>
> nano 버전 기준 Benchmark 결과

최신 `YOLO` 모델은 FLOPs(연산량)과 Params(파라미터 수)가 더 적으면서도 AP(Average Precision)가 높습니다.<br/>
이는 리소스 대비 성능이 우수함을 의미합니다.<br/>
- 참고 1. [YOLOv12: Attention-Centric Real-Time Object Detectors](https://arxiv.org/pdf/2502.12524)<br/>
- 참고 2. [Ultralytics YOLO Docs](https://docs.ultralytics.com/ko/models/yolo11/)<br/>
<br/>

특히, 이 프로젝트에서는 사람의 얼굴을 중점으로 학습한 `YOLO-face` 모델을 사용합니다.<br/>
- 참고 3. [yolo-face](https://github.com/YapaLab/yolo-face)<br/>
<br/>

위 모델(`YOLO-face`)을 사용해서 `local`, `fastapi`, `gradio` 환경에서 정상 구동 여부를 테스트합니다.<br/>
<br/>
단, 모델 변경만으로는 오탐률은 낮아질 수 있으나 **오탐시의 대응 방안은 별도의 고려가 필요합니다.**<br/>

## 모델별 추론 시간
테스트에는 최신 버전의 `YOLO nano` 모델을 사용했습니다.<br/>
비교 대상은 `yolov11n-face`와 `yolov12n-face`이며, 후자는 `.pt`와 `.onnx` 두 형식으로 측정했습니다.<br/>

### 1. Test 환경

|항목|내용|
|--|--|
|Device|Macbook Pro(M3 Pro, 12-core CPU, 18GB RAM)|
|Architecture|ARM64 (Apple Silicon)|
|OS|Mac OS 14.4 (23E214)|
|Python|3.11.10|

### 2. `yolov11n-face.onnx`
![yolov11 infer time](/assets/images/yolov11-onnx-infer_time.png)<br/>
평균 추론(처리)시간 약 **25ms**<br/>

### 3. `yolov12n-face.pt`
![yolov12 pt infer time](/assets/images/yolov12-pt-infer_time.png)<br/>
평균 추론(처리)시간 약 **50ms**<br/>
Apple Silicon 환경에서는 CUDA 최적화가 부족하므로 PyTorch(`.pt`)모델의 추론 속도가 느리게 측정됩니다.<br/>

### 4. `yolov12n-face.onnx`
`yolov12n-face`의 경우, `.onnx` 형식이 공식 배포되지 않아 [yolo-face](https://github.com/YapaLab/yolo-face)의 레퍼런스를 따라 아래와 같이 직접 변환했습니다.<br/>

```python
from ultralytics import YOLO

m = YOLO("./models/yolov12n-face.pt")
m.export(format="onnx", opset=21, simplify=True, dynamic=True, imgsz=640)
```

![yolov12 infer time](/assets/images/yolov12-onnx-infer_time.png)<br/>
평균 추론(처리)시간 약 **22ms**<br/>

> 결론적으로 동일 환경 기준 `.onnx`형식이 `.pt`대비 **약 2배 빠른 추론 속도**를 보였습니다.

## 폴더구조
```
project_root/
│
├── README.md
├── assets/
├── requirements.txt
└── src/
    ├── models/
    │   ├── yolov11n-face.onnx
    │   ├── yolov12n-face.onnx
    │   └── yolov12n-face.pt
    ├── test_fastapi.py
    ├── test_gradio.py
    └── test_local.py
```

## 실행 방법

### 1. 가상환경 설정

```
python -m venv {venv_name}
source {venv_name}/bin/activate
pip install -r requirements.txt
cd src
```

#### 2. Local Test

```
python test_local.py
```

웹캠 영상이 실행됩니다.<br/>
종료는 `esc` 키를 누릅니다.

#### 3. FastAPI Test

```
uvicorn test_fastapi:app --reload
```
`localhost`인 http://127.0.0.1:8000 에 접속하여 확인할 수 있습니다.<br/>
종료는 `CTRL+C` 입니다.

#### 4. Gradio test

```
python test_gradio.py
```

웹캠 아래 `녹음`버튼을 클릭하면 추론이 시작됩니다.<br/>
Public URL보다는 Local 접속을 권장합니다. (Public URL 사용 시 병목 발생 가능)

# 실험 환경

| **OS** | Windows 11 |
| --- | --- |
| **Python version** | 3.10.18 |



# 실행 방법

1. Git repo 복사

```
git clone https://github.com/ssun-g/ai_assignments.git
```

1. 패키지 설치

```
cd ai_assignments
pip install -r requirements.in
```

1. 학습 데이터 다운로드(이후 폴더 구조는 아래와 같아야 함)

```
├─checkpoints/
├─data/  # 학습 데이터
│  └─train/
│      ├─apple/
│      ├─cherry/
│      └─tomato/
├─utils/
├─compare_models.ipynb
├─EDA.ipynb
├─onnx&quant.ipynb
├─requirements.in
└─train.ipynb
```

### 실행 순서

1. `EDA.ipynb`: EDA, 데이터 시각화
2. `train.ipynb`: 모델 학습 및 평가
3. `onnx&quant.ipynb`: 학습 완료된 모델 ONNX 변환 및 INT8 양자화 수행
4. `compare_models.ipynb`: 모델 성능 비교(전이 학습 모델, ONNX 변환 모델, INT8 양자화 모델)



# EDA.ipynb - EDA (데이터 시각화)

### 분류 가능성 검증

![Image](https://github.com/user-attachments/assets/6490e24f-d8c8-43be-b04d-a9cd0993c8f1)

- **KMeans Clustering**
    - 사전 학습 된 `ConvNet` 모델로 추출한 feature를 기반으로 KMeans 클러스터링 수행(`apple`, `lychee`, `banana`, `cherry`, `orange` 총 5개 군집)
    - 추출한 feature를 PCA로 차원 축소하여 시각화
    - 클래스 간 feature 분포가 분리되는지 시각적으로 확인하여 분류 작업 수행 가능성 확인

- **결론**
    - 시각적으로 확인 결과 총 3개의 군집으로 나눠지는 것 확인
    - `apple`, `cherry`, `tomato` 3개의 클래스로 전이학습 할 것이므로 분류가 가능할 것으로 예상

### 클래스 별 데이터 수

![Image](https://github.com/user-attachments/assets/47bf410f-d5b7-445e-92d0-992da5dd35f5)

### 문제

- apple: 127, cherry: 127, tomato: 50개의 데이터가 존재함
- 데이터 불균형 문제 확인 가능

### 해결 방법

- 모델 학습 시 tomato 클래스에 속하는 데이터는 이미지 증강을 적용하여 학습



# train.ipynb - 모델 학습 및 평가

### 모델 학습 과정

1. 전체 데이터를 Train, Valid, Test로 분할. 비율은 70:15:15
2. 사전 학습 모델 attention 시각화(Grad-CAM)
3. 전이 학습(validation dataset의 f1 score가 가장 높은 weight 저장)
4. 학습시, 데이터의 양이 적은 “tomato” 클래스의 데이터만 증강 적용

```python
# utils/datasets.py (Line 31 ~ 35)

# 데이터가 적은 클래스만 증강 적용
if label == self.minority_class:
    image = self.augmented_transform(image)
else:
    image = self.basic_transform(image)
```

1. Test dataset에 대해 시각화 및 모델 평가
2. 학습 완료된 모델의 attention 시각화 (Grad-CAM)

### 평가 지표: F1 score(macro)

- EDA를 통해 데이터 불균형 확인
- 따라서 모델의 평가 지표는 정확도 보다 F1 score가 적합하다고 생각하여 평가 지표로 선택

### Grad-CAM 시각화 결과

- 모델 학습 이전

![Image](https://github.com/user-attachments/assets/23f9cc34-5adc-46e3-a8df-8dc32e3c7b4c)

- 모델 전이 학습 이후

![Image](https://github.com/user-attachments/assets/c135f6ae-2437-4ae8-abad-140647fdf5c3)



# onnx&quant.ipynb - 모델 ONNX 변환 및 INT8 양자화

### ONNX 변환 시 주의사항

- 배치 사이즈의 크기를 유동적으로 입력 받기 위해 `dynamic_axes` 추가

```python
torch.onnx.export(model,
                  dummy_input,
                  "conv_net.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}})  # 0번째 차원 dynamic
```

### INT8 양자화 시 주의사항

- 데이터의 수가 적으므로 모든 데이터를 calibration data로 사용

```python
dataset = datasets.ImageFolder("./data/train/", transform=transform)  # 전체 데이터
calibration_loader = DataLoader(dataset, batch_size=CFG['bs'], shuffle=False)
data_reader = ImageFolderDataReader(calibration_loader)

quantize_static(model_input="conv_net.onnx",
                model_output="conv_net_int8.onnx",
                calibration_data_reader=data_reader,  # calibration data로 전체 데이터 사용
                quant_format=QuantFormat.QOperator,
                weight_type=QuantType.QInt8)
```



# compare_models.ipynb - 모델 결과 비교

### 최종 모델

`./checkpoints/conv_net_int8.onnx`

### 결과 비교

![Image](https://github.com/user-attachments/assets/517f3cc9-5bcd-42e6-866d-d0e880f39f70)
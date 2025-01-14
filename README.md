# 기계학습과 응용 기말프로젝트

## 프로젝트 소개

아이를 키우는 처음 키우는 부모나 청각 장애을 가지고 있는 부모들은 아이가 무엇때문에 우는지 정확하게 파악하지 못할 때가 있습니다. 특히 청각 장애를 가지고 있는 경우 울음소리를 아예 듣지 못할 수도 있습니다. 이를 해결하기 위해 아기 울음소리 감지 및 분류 프로그램을 만들었습니다.

---

## 디렉토리 구조

### 0. requirements.txt

프로젝트에 필요한 python패키지를 명시한 파일입니다.


### 1. data

아이 울음소리에 대한 학습 데이터셋이 있으며 데이터들은 awake, diaper, hug, hungry, sad, sleepy, uncomfortable으로 분류하고 있습니다. 또한 각 폴더 안에는 **.wav 형식**의 오디오 파일이 포함되어 있습니다. 


### 2. test_audio

테스트할 파일을 저장하는 디렉토리입니다. 이 디렉토리 안에 있는 파일들은 프로그램이 실행될 때 테스트하게 됩니다. (.wav형식의 오디오 파일만 지원)


### 3. models

학습된 모델을 저장하는 디렉토리입니다. (baby_cry_model_crnn.keras: CRNN 모델 파일)


### 4. scripts

프로젝트의 주요 기능들을 모듈화한 스크립트들이 있습니다.

1. data_processing.py : 데이터를 불러오고 전처리를 수행
2. crnn_model.py : Convolutional Recurrent Neural Network(CRNN)에 대하여 정의를 내려 모델의 작동 방식을 포함
3. predict.py : 학습된 모델을 이용하여 test_audio 디렉토리에 있는 데이터에 대해 예측 수행
4. utils.py : 디렉토리 생성, 파일 경로 처리 등 여러 보조 함수 포함


### 5. baby_cry.py

이 프로젝트의 메인 실행 파일

---

## 주요 기능
- **오디오 전처리**: `.wav` 형식의 오디오 파일을 처리하고 MFCC(Mel-Frequency Cepstral Coefficients) 특징을 추출.
- **딥러닝 모델**: CNN과 RNN을 결합한 CRNN 모델을 사용.
- **실시간 예측**: 새로운 오디오 파일을 가지고 결과를 제공.
- **데이터셋 확장 가능**: 사용자 데이터셋을 추가하여 모델 정확도를 높임.

---

## 향후 개선 사항
- 현재는 .wav 확장자만 가능해 다른 확장자를 가진 파일에 대해서는 프로그램에 넣을 수 없습니다. 그러므로 이후에 다른 확장자 또한 가능하게 개선하고 싶습니다.
- 현재 프로그램에서는 소리를 감지하는 기능이 존재하지 않아 청각장애인 부모이면 울음소라 자체를 듣지 못할 가능성이 높습니다. 그러므로 이 프로젝트의 목표에 도달하기 위해 소리를 감지하고, 이 소리가 아기의 울음소리인지 파악하고, 마지막에 모델에 넣어 결과를 예측하는 프로그램을 만들어야 합니다.
- 몇 가지 종류의 울음소리 (예를 들어 hungry)에 대해서는 맞힐 확률이 높지만 어떤 종류는 틀리는 경우가 많습니다. 어떻게 하면 더 정확한 모델을 만들 수 있을까 고민을 해봐야 할 것 같습니다.

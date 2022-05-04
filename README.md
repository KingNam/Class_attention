# Mediapipe 기반 Hpe(Head Position) 와 졸음인식

안녕하세요. 통계학 전공한 송영남이라고 합니다.

Mediapipe의 Face Mesh 코드를 기반으로 얼굴이 어딜 보고 있는지와 졸고 있는지 아닌지를 판단하는 프로그램을 개발해보았습니다.

첫 프로젝트 업로드라 부족한 점이 많을것으로 예상됩니다. 피드백 주시면 빠르게 반영하겠습니다. 감사합니다. 

참고자료부터 첨부하겠습니다.

HPE 참고자료: https://youtu.be/-toNMaS4SeQ

Eye 참고자료: https://github.com/Asadullah-Dal17/Eyes-Position-Estimator-Mediapipe

Mediapipe 참고자료: https://google.github.io/mediapipe/solutions/face_mesh

HPE 영상에 첨부된 코드를 기반으로 v1을 제작하였고 OOP를 배우고 난 뒤, Detection 된 Mesh에서 정보를 받아와 처리하였습니다.

## Attention_v1

mediapipe의 최장점은 세팅이 간단하다는 점입니다. 

meidapipe를 처음 다뤄보신다면 가상환경을 설치해 코드를 실행하시면 됩니다.

``` conda create -n my_mediapipe```

``` conda activate my_mediapipe```

``` pip install mediapipe```

의 명령어를 cmd 에서 실행하신 후, 시작합시다

### ultimate.py

Attention_v1 의 본체 입니다. 

함수를 하나씩 설명드리겠습니다.

```python
def landmarkdetection(img, result, dimension=2, draw=False) : landmark 좌표를 출력합니다. (Pixel 좌표계)
# img: 이미지를 받는 인자 입니다
# result: mediapipe의 face_mesh에서 계산되어 나온 결과 값입니다. {x: 0.~~, y: 0.~~, z: 0 ~~~} 형태의  Frozen Set 입니다.
# dimension: 차원 수를 결정합니다. 2 차원 혹은 3차원을 선택할 수 있게 해놓았습니다. 
# draw: 추출된 점을 그립니다. (저는 실제 사용한적이 없습니다...)
```

```python
def eculideanDistance( ~~~) : 점과 점 사이의 직선거리를 구합니다.
```

```python
def BlinkRatio( ~~~): EAR(Eye Aspect Ratio)를 반환합니다. 아래 수식의 결과를 반환합니다.
```

![EAR](https://user-images.githubusercontent.com/48468043/166609866-6122799c-952a-4261-8775-1832866f9f4e.jpg)


```python
HpeEstimate: 코와 눈썹, 얼굴 윤곽 landmark를 기반으로 얼굴 방향에 대한 축을 생성합니다. 
```

로직이 꽤 복잡한데, Pixel 좌표계를 월드 좌표계로 반환하여 점을 생성합니다. (정확하진 않음)

### utils.py

제가 직접만든 파일은 아니라 모든것을 알진 못하지만, 함수이름이 직관적으로 짜여있습니다.

주로 글자를 만들거나 그림을 그려낼 때 사용합니다.

### Attetion_fun.py

수업시간을 가정한 환경에서 학생의 집중도를 체크하는 함수를 만들었습니다. 

졸음, 자리이탈, 집중 x (화면을 보고 있지 않음) 의 점수를 매겨 최종 집중도 점수를 산출합니다.

### Attention.py

Attention_fun.py 를 객체지향으로 바꾸었습니다. 한 수업에 여러명의 학생이 들어오는 경우를 고려해 객체지향 코딩하였습니다.








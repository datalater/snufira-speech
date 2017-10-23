
ⓒ JMC 2017


---

## 20171023 3주차

**sound wave**. 전파와는 다르다.
빛이나 라디오 주파수는 전자파이다.
sound wave는 공기 중의 질소나 산소를 미는 것이다.
그래서 sound pressure wave라고도 한다.
sound wave는 전자기파보다 느리므로 속도 차이가 엄청나게 난다.

오늘은 말(talk)을 어떻게 produce하는가를 이야기한다.
허파에 공기가 들어가고, 그 위에 있는 성대(vocal cord)에서 진동이 생기고, 진동은 성도로 올라가서 입 밖으로 나온다.
목소리에는 유성음과 무성음이 있다.
가령 '스' 같은 발음은 떨리지 않는 무성음이다.
유성음은 진동이 있는데, 이 진동의 주파수에 따라 높은 음이나 낮은 음의 소리가 나면서 음계가 달라진다.
고음을 잘 내는 사람은 그만큼 성대(vocal cord)를 잘 컨트롤할 수 있다는 뜻이다.

**Excitation = vocal cord**.
Excitation과 Excitation으로 vocal tract(=관)을 소리를 구분할 수 있다(?).

사람은 vocal tract이 길다.
피리의 길이가 긴 것과 같은 이치이다.
vocal tract이 긴 만큼 다양한 소리를 낼 수 있다.
개는 vocal tract이 매우 짧다.

Pitch: 도레미파솔라시도 음계.
같은 말로 fundamental frequency라고도 한다.
시간축에서 어떤 소리를 냈을 때 sound wave의 반복되는 간격이 Pitch이다.
낮은 음의 소리를 내면 Pitch가 늘어난다.
발음에 따라서 달라지는 것은 wave form이다.

음성의 2가지 요소.
lid를 진동시키는 주파수를 컨트롤 한다.
그 주파수가 pitch이다.
소리가 전달되는 파이프의 길이를 조절해서 wave form을 필터링한다.
vocal cord는 도레미파솔라시도 같은 pitch(fundamental frequency)를 결정한다.
vocal tract(mouth 포함)는 어떤 발음일지 음소를 결정한다.

**Phonemes vs. Phones**.
사람의 발음은 피아노처럼 딱딱 떨어지게 되지 않는다.
Phonemes: '아에이' 같은 것. 영어에는 42개가 있다.
Phone: 수천 개.

Phoneme을 classify하는 법:
+ Vowel: Front, Mid, Back. 혀의 위치에 따라 구분된다.
+ Consonant : 마찰음, 등

**Discrete-time Model for Speech Production**.
bandpass 3개 이상이 되는 digital filter를 만든다.
vocal tract 역할을 하는 것이 digital filter이다.
사람의 목소리와 비슷하려면 digital filter가 1초에 50번 이상 변해야 한다.

**Feature Extraction**.
디지털 필터의 모양을 아는 것.
음성 인식에서는 pitch-period는 중요하지 않다.
유성음이나 무성음이냐가 중요하다.
음성인식에서는 디지털 필터의 주파수를 아는 것이 중요하다.
그것을 feature extraction이라고 한다.

사람의 목소리가 vocal tract filter를 거치면 복잡한 wave form이 나온다.

vocal cord를 10Msec(=100Hz)로 소리 낸다고 해보자.
입에서 나오는 소리는 같은 주기이지만 다른 형태로 나온다.
vocal cord를 FT한 주파수를 digital filter를 통과시킨 주파수에서 가장 높은 주파수가 무엇인지 아는 것이 중요하다.


**음성 인식 Feature Extraction**.
+ MFCC
+ MEl Frequency Break Spo..

**Mel Scaled-Filter Bank**.
높은 주파수로 갈수록 넓어진다.
사람의 귀에서 원리를 따온 것.
낮은 주파수로 갈수록 filter bank가 좁아진다.
filter bank가 좁다는 것은 그만큼 섬세하게 잡아낸다는 것을 뜻한다.
filter bank는 20몇개로 구성된다.
사람의 달팽이관에 있는 hair cell(신경세포)과 동일하게 개수를 맞추기 위해서.
각 filter bank에 있는 signal을 서로 비교해서 어느 filter bank의 signal이 더 큰지를 따진다.

**Cepstrum(캡스트럼)**.
피치의 주파수에서 이상한 것을 제거하는 것.
곱셈을 없애려면 log를 취한다.
log(A*B) = log(A) + log(B).
A = 피치 주파수. 빨리 변하는 성분.
B = vocal tract이 만드는 성분. 천천히 변하는 성분.
여기에 low-pass filter를 적용하면 느리게 있는 성분(B)만 골라낼 수 있다.
즉, 피치 성분을 없애서, 여자 목소리같은 고음이나 남자 목소리 같은 저음의 목소리 변화는 다 없애버리고 중저음의 목소리만 남긴다.

Excitation signal.
100Hz 마다 반복되는 시그널.

w는 주파수.
`S(w)=E(w).H(w)`


Ex.
+ x(n) time domain signal.
  + 피치가 40msec = 250Hz.
  + 여자 목소리일 가능성.
+ windowed signal
  + 가장 자리에는 작은 값을 곱하고 가운데에는 큰 값을 곱한다.
  + 그러면 가운데 모양만 살아남는다.
+ FT | x(w) =dft(x(n))
  + ...
+ Log 스케일로 보기
  + 3개의 filter bank를 지나는 speech라는 것을 알 수 있다.
+ C(n) = iDft(Log(|x(w)|)
  + ...


**Feature Extraction Using MFCC**.

Input Speech =>

1. Framing and windowing : time-domain에서 가운데 값만 남긴다. 목소리가 '가나다라마바사'일 때 대략 20msec 동안은 목소리가 동일한 음이 난다고 생각을 한다. 그림을 50msec로 넘기면 사람은 움직이는 그림으로 인식하는 것과 같은 원리이다. 목소리를 나눌 때도 10msec~20msec로 나누면 목소리가 stable하다. 그렇게 잘라서 spectrum analysis를 한다. 1초짜리 음성을 분석할 때는 20msec 윈도우를 50번 해야 한다.
2. Fast Fourier Transfrom : 주파수별 성분을 표시한다.
3. Absolute value :
4. Mel scaled-filter bank : 완만한 선을 만든다. 방법은 다음과 같다. k=1, ... 20. k값에 따라 그 구간의 값을 측정한다. Neural Network에 집어 넣으면 인식 결과가 "Hello"라고 생긴다. 짧은 간격의 골(계곡의 골)을 없애서 신호를 부드럽게 바꾼다. 이를 MFCC라고 한다. pitch를 없애면 인식이 더 잘 된다. 피치의 영향을 없애는 게 MFCC인데 지금은 training data가 많아서 MFCC를 굳이 쓰지 않는다. ex. pitch 간격 = 100hz 사이의 골.
5. Log :
6. Discrete cosine transform :

=> Feature veoctors

음성인식을 할때는 4번 이후의 값이나 6번 이후의 값을 사용한다.
훈련데이터가 적을 때는 6번 이후의 값을 사용하는 게 유리하다.
4번을 하면 훈련데이터가 줄어드는 셈.
Neural Network 덕분에 훈련데이터를 쉽게 만드는 방법이 많아졌다.

Windowing에서 제일 많이 쓰는 게 HAMMING WINDOW이다.

**Feature Vector**.
+ MFCC : 13차 계수를 얻는다. Mel frequency를 지나면 계수가 20 몇 개로 줄어든다. FT를 한다.
+ Delta MFCC : C(n)-C(n-1) (C=계수. C(n)= frame 된것)
+ Delta Delta MFCC : DD(n) = D(n) - D(n-1)
  + 말을 하면 반사되어 오는 게 문제가 된다.
  + 반사되는 주파수는 어떻게 없앨까.
  + 사람의 vocal tract 길이에 비해 방의 크기는 훨씬 더 크다.
  + 거인의 vocal tract가 일반인에 비해 500배처럼 엄청 더 크다고 하면 말을 알아듣기 힘들 것이다. 왜냐하면 매우 낮은 주파수일 것이기 때문에.
  + 반사된 주파수를 없애는 것이 Delta 또는 Delta Delta이다.
  + Delta는 일종의 filter라고 볼 수 있다.
  + 덧셈은 ..-pass filter, 뺄셈은 ..-pass filter `10.23.11:46 rec ~37:20~`

39차 => GMM or HMM => 음성인식

Mel을 집어 넣어서 => 음성인식

**음성인식 방법**.
1. MFCC + GMM + HMM
2. Filter bank + DNN + HMM
3. Filter bank + RNN(CTC) + 간단한 decoding

1번 방식은 점점 사라지는 중. 2번째는 실용적으로 가장 인식률 높음. 그런데 슬슬 없어짐. 3번째가 떠오르는 방법.





---

## 20171018 2주차

이전 시간에는 filtering을 해서 원하는 신호만 끄집어 내는 것을 했다.
이번 시간에는 어떤 신호를 없애기보다는 신호의 주파수를 분석한다.

Sine wave의 주파수.
주파수는 1초에 몇 번 진동하느냐를 의미한다.
사람의 귀는 20~20,000Hz(20Khz)를 인식할 수 있다.
1초에 20번 진동하는 소리부터 20만 번 진동하는 소리를 들을 수 있다.
1초에 20번 진동하는 소리를 Time Domain(시간축)으로 시각화하면 1초 동안 20번 반복되는 패턴(sine wave 형태)을 볼 수 있다.
Time Domain 상에서 반복되는 패턴 하나가 시작해서 끝나기까지 걸리는 시간을 주기라고 한다.

> **Note**: 신호에서 반복되는 패턴을 cycle이라고 한다.

+ 주기(T) = 1Msec = 1/1000sec = 1000Hz = 주파수(freq)
+ 주기(T) = 5Msec = 1/200sec = 200Hz = 주파수(freq)
+ 주파수(freq) = $\frac{1}{T} Hz$
+ angular freq = $\frac{1}{T} * 2 \pi \ radian$

> **Note**: $2 \pi$는 한 바퀴를 의미한다.

Fourier Transfrom.
original 신호는 여러 가지 신호가 합쳐진 것으로 볼 수 있는데, 여러 가지 신호는 original 신호의 정수배로 이루어져있다..
피아노 C음 520Hz를 시각화하면 복잡한 모양이다.
거기에는 520Hz, 1040Hz, 1560Hz가 섞여 있다.
FT(푸리에 트랜스폼)이 하는 일은 복잡한 모양의 original 신호(기본 주파수)를 1x, 2x, 3x의 성분의 크기를 알아내는 것이 목적이다.
그것을 Harmonics(고조화)라고 한다.

> **Note**: 특정 주파수만 가진 단일 음을 들려주면 사람이 듣기에 좋지 않다. 장난감 피아노 소리가 듣기 좋지 않은 이유도 같은 맥락이다.

FT의 식은 적분식이다.
적분을 하는 이유는 기본 주파수에 어떤 주파수의 성분이 많이 들어 있는지 알 수 있기 때문이다.
왜냐하면 같은 주파수를 적분하면 값이 커지고, 다른 주파수를 적분하면 값이 작은 성질이 있기 때문이다.
공식에서 $f(t)$와 $w$의 공진 주파수를 찾는 방법이 적분인 것이다.

Discrete Fourier Transform.





---

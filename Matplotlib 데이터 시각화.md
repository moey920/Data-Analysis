# Matplotlib
- 파이썬에서 데이터를 그래프나 차트로 시각화할 수 있는 라이브러리

## 그래프 그려보기
```
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]
plt.plot(x, y) # (1,1), (2,2), (3,3), (4,4), (5,5)를 연결한 직선이 그려진다.
```

- State Machine Interface : 자동으로 figure와 ax 그림
```
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]
plt.plot(x, y)
plt.title("First Plot") # 그래프 영역 위에 타이틀 설정
plt.xlabel("x") # x 축의 라벨 지정
plt.ylabel("y") # y 축의 라벨 지정(글자가 누어져있다.)
```

- 다른 방식으로 그래프 그려보기(Object Oriented Interface) : 객체 기반 스타일(figure와 ax를 수동으로 생성)
- 취향에 따라 SMI와 OOI 중 선택하여 그린다.(OOI가 더 명시적)
```
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]
fig, ax = plt.subplots()
ax.plot(x, y) # 데이터 집어넣기
ax.set_title("First Plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
```

## Matplotlib구조
- Figure : 도화지, 라인 그래프와 산점도 그래프 등을 모두 포함하는 가장 큰 화면
- Axes : 각 그래프를 포현(그래프 화면)
- Title : set_title
- y label : set_ylabel
- x label : set_xlabel
- Line (line plot) : 각 점이 선으로 이루어진 그래프
- Marker (scatter plot) : 각 점이 화면위에 찍혀있는 그래프(점 그래프)
- Grid : 그래프 내 격자모양(수정 가능)
- Major tick : 각 라벨의 큰 눈금
- Minor tick : 각 라벨의 작은 눈금
- Legend : 범례, 각 그래프의 종류를 나타내는 문구

- 저장하기
```
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]
fig, ax = plt.subplots() # 전체 도화지(figure)와 해당 그래프(ax)를 정의하는 함수
ax.plot(x, y)
ax.set_title("First Plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.set_dip(300) # 전체 도화지를 저장해야하니 fig를 이용한다. 1인치 제곱당 몇 도트까지 들어갈 수 있는지 지정. dot per inch. 300정도 지정하면 출력물에 대해서는 웬만해선 제대로 보인다. 
fig.savefig(”first_plot.png”) # 파일명과 확장자를 명시하여 savefig() 함수를 이용한다.
```

- 여러개 그래프 그리기
```
x = np.linspace(0, np.pi*4, 100) # 0부터 4파이까지 100개의 구간으로 나눈 array 객체
fig, axes = plt.subplots(2, 1) # 세로축이 2(그래프가 2개), axes[0], axes[1]이 나온다.
axes[0].plot(x, np.sin(x)) # x 축에 x값을, y 축에 x에 해당하는 sin()값을 넣는다.
axes[1].plot(x, np.cos(x))
```

```
from elice_utils import EliceUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
elice_utils = EliceUtils()

x = [1, 2, 3, 4, 5]
y = [5, 6, 7, 8, 9]
# 그래프를 그리는 코드 작성
fig, ax = plt.subplots(2, 1)
ax[0].plot(x, y)
ax[1].plot(y, x)
# ax.plot(x, y)
ax[0].set_title("First Plot")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[1].set_title("Second Plot")
ax[1].set_xlabel("y")
ax[1].set_ylabel("x")

# elice에서 그래프를 확인
fig.savefig("first_plot.png")
elice_utils.send_image("first_plot.png")
```

## Matplotlib 그래프들
- Lineplot
```
fig, ax = plt.subplots()
x = np.arange(15)
y = x ** 2
ax.plot(
    x, y,
    # 세가지 옵션
    linestyle=":", 
    marker="*", 
    color="#524FA1"
)
```
> Linestyle

```
x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(x, x, linestyle="-") # linestyle="solid" 도 작동한다.
# solid(선)
ax.plot(x, x+2, linestyle="--")
# dashed
ax.plot(x, x+4, linestyle="-.")
# dashdot
ax.plot(x, x+6, linestyle=":")
# dotted(점선)
```

> Color

```
x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(x, x, color="r") # rgbcmyk 가능
ax.plot(x, x+2, color="green") # 풀네임도 가능
ax.plot(x, x+4, color='0.8') # 0~1 사이 값, 문자열로 들어가면 grayscale로 들어간다(회색조)
ax.plot(x, x+6, color="#524FA1") # rgb 16진수 코드도 가능
```

> Marker : 점선에 표시할 마크

```
x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(x, x, marker=".") 
ax.plot(x, x+2, marker="o")
ax.plot(x, x+4, marker='v') # 삼각형
ax.plot(x, x+6, marker="s") # Square 네모박스
ax.plot(x, x+8, marker="*")
```

> 축 경계 조정하기(그래프 자체의 옵션) : 어디서 시작하고, 어디서 끝나는지

```
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x))
ax.set_xlim(-2, 12) # x축이 -2~12까지 그려진다
ax.set_ylim(-1.5, 1.5) # y축이 -1.5~1.5까지 그려진다
```

> 범례

```
fig, ax = plt.subplots()
ax.plot(x, x, label='y=x') # 하나의 그래프에 두 선을 표현한다. x=y인 그래프
ax.plot(x, x**2, label='y=x^2') # y = x^2 인 그래프, 각 그래프에 label값을 준다(범례)
ax.set_xlabel("x") # 그래프의 x lable 설정
ax.set_ylabel("y")
ax.legend( # 범례 옵션에 표시한 내용이 그래프에 반영된다. 
    loc='upper right', # 그래프 내에서 범례 위치. 오른쪽 상단 / "lower left", "center" 등도 가능
    shadow=True, # 범례 상자에 shadow 기능 추가
    fancybox=True, # 범례 상자 모서리를 둥글게 만들기
    borderpad=2 # 범례 상자 내의 border(범례 상자 크기를 조정)
)
```

```
from elice_utils import EliceUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
elice_utils = EliceUtils()

#이미 입력되어 있는 코드의 다양한 속성값들을 변경해 봅시다.
x = np.arange(10)
fig, ax = plt.subplots()
# Line plot 옵션 조절해보기
# Markers {'.' : 점, ',' : 픽셀, 'o' : 원, 's,p' : 사각형, 'v,<,^,>' : 삼각형, '1,2,3,4' : 사각선, 'H,h' : 육각형}
ax.plot(
    x, x, label='y=x',
    marker='H',
    color='blue',
    linestyle='--'
)
ax.plot(
    x, x**2, label='y=x^2',
    marker='p',
    color='g',
    linestyle='-.'
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(
    loc='upper left',
    shadow=True,
    fancybox=True,
    borderpad=3
)

fig.savefig("plot.png")
elice_utils.send_image("plot.png")
```

## Scatter(산점도) 

```
fig, ax = plt.subplots()
x = np.arange(10)
ax.plot(
    x, x**2, "o", # x,y값, 점의 형태 - line plot이 아닌 Scatter로 명시
    markersize=15, # marker(원)의 크기
    markerfacecolor='white', # marker의 내부 색
    markeredgecolor="blue" # marker의 테두리 색
)
```

- 사이즈와 컬러를 각각 지정할 수 있다.
```
from elice_utils import EliceUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

elice_utils = EliceUtils()

fig, ax = plt.subplots()
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.randint(0, 100, 100)
sizes = 500 * np.pi * np.random.rand(50) ** 2

ax.scatter(
    x, y, # 점 중앙의 좌표
    c = colors, s = sizes, alpha = 0.3 # alpha는 투명도, 점들이 겹쳐 보일 수 있게 함
)

fig.savefig("plot.png")
elice_utils.send_image("plot.png")
```

## Bar & Histogram

### Bar plot
```
x = np.arange(10)
fig, ax = plt.subplots(figsize=(12, 4)) # 가로 12, 세로4 크기의 사이즈 지정
ax.bar(x, x*2) #x, y 값
```

- 축적그래프 : 누적하여 그리는 Bar plot

```
x = np.random.rand(3)
y = np.random.rand(3)
z = np.random.rand(3)
data = [x, y, z]
fig, ax = plt.subplots()
x_ax = np.arange(3) # 0,1,2 가로축 인덱스라고 생각하면 된다.
for i in x_ax:
    ax.bar(x_ax, data[i], # data[i] = [x,y,z], x_ax = x축, data[i] = y축
    bottom=np.sum(data[:i], axis=0)) # 이전 데이터로부터, y축으로 데이터를 누적한 곳에서 시작하여 그린다.
ax.set_xticks(x_ax) # xticks를 0,1,2로 설정
ax.set_xticklabels(["A", "B", "C"]) # 0,1,2를 "A,B,C"로 수정
```

### Histogram (도수분포표)
```
fig, ax = plt.subplots()
data = np.random.randn(1000) # 표준정규분포에서 1000개를 뽑는다.
ax.hist(data, bins=50) # bins : 막대기를 몇 개 사용할 것인지.
```

```
from elice_utils import EliceUtils
elice_utils = EliceUtils()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib 의 pyplot으로 그래프를 그릴 때, 기본 폰트는 한글을 지원하지 않습니다.
# 한글 지원 폰트로 직접 바꾸어주면 한글을 사용하실 수 있습니다.
import matplotlib.font_manager as fm
fname='./NanumBarunGothic.ttf'
font = fm.FontProperties(fname = fname).get_name()
plt.rcParams["font.family"] = font


x = np.array(["축구", "야구", "농구", "배드민턴", "탁구"]) # 선호종목
y = np.array([18, 7, 12, 10, 8]) # 각 종목을 선호하는 학생의 수

z = np.random.randn(1000)

fig, axes = plt.subplots(1, 2, figsize=(8, 4)) # 하나의 도화지(figure)에 1*2의 모양으로 그래프를 그리도록 합니다, 그래프의 사이즈는 8x4입니다.

# Bar 그래프
axes[0].bar(x, y)
# 히스토그램
axes[1].hist(z, bins = 50)

# elice에서 그래프 확인하기
fig.savefig("plot.png")
elice_utils.send_image("plot.png")
```

## Matplotlib with Pandas : Numpy데이터가 아닌 Series나 DataFrame을 이용해서 그래프 그리기

- Series 데이터를 x,y 값으로 활용
``` 
df = pd.read_csv("./president_heights.csv")
fig, ax = plt.subplots()
ax.plot(df["order"],df["height(cm)"], label="height")
ax.set_xlabel("order") # 몇 번째 대통령인지
ax.set_ylabel("height(cm)") # 키가 몇인지
ax.legend()
```

- 불 포켓몬과 물 포켓몬의 분표 살펴보기
```
from elice_utils import EliceUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

elice_utils = EliceUtils()

df = pd.read_csv("./data/pokemon.csv")

fire = df[
    (df['Type 1']=='Fire') | ((df['Type2'])=="Fire") # Type1,2 중에 불 속성이 있는 포켓몬의 값을 가져옴
    ]
water = df[
    (df['Type 1']=='Water') | ((df['Type2'])=="Water")
    ]
fig, ax = plt.subplots()
ax.scatter(fire['Attack'], fire['Defense’], # x,y값
            color='R', label='Fire', marker="*", s=50) # scatter 속성 설정
ax.scatter(water['Attack'], water['Defense’],
            color='B', label="Water", s=25)
ax.set_xlabel("Attack")
ax.set_ylabel("Defense")
ax.legend(loc="upper right")
```

- 토끼와 거북이 경주 결과 그래프 그리기
```
from elice_utils import EliceUtils
from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = 'NanumBarunGothic'

elice_utils = EliceUtils()


# 아래 경로에서 csv파일을 읽어서 시각화 해보세요
# 경로: "./data/the_hare_and_the_tortoise.csv"
def main() :
    df = pd.read_csv("./data/the_hare_and_the_tortoise.csv")
    df.set_index("시간", inplace=True) # 기존의 index와 시간 data가 똑같이 중복되기 때문에 시간을 기준으로 index를 변경해준다
    # print(df)
    # rabbit = df[df["토끼"]]
    # 그래프 그리기
    fig, ax = plt.subplots()
    ax.plot(df["토끼"], label="토끼")
    ax.plot(df["거북이"], label="거북이")
    ax.legend()
    
    fig.savefig("plot.png")
    elice_utils.send_image("plot.png")

if __name__ == "__main__" :
    main()
```

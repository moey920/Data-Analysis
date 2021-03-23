# 모듈

- 모듈은 특정 목적을 가진 함수, 자료의 모임입니다.
- .py로 만든 파일은 모두 모듈로 활용이 가능합니다.
- import를 이용하면 모듈을 불러올 수 있습니다 .as(alias)를 활용하면 긴 이름의 모듈을 짧게 정의하여 사용합니다.

## 모듈 만들기
```
import random
# random 모듈 불러오기
import numpy as np
# numpy 모듈을 불러오고 ‘np’ 두 글자로 사용할 수 있게 설정
```

### function vs attribute

- function (default setting)
    - plus()
    - print()
    - value_count()

- attribute
    - object.shape
    - object.index
    - 소괄호를 쓰지 않는 attribute는 따로 인자를 주지 않고 객체의 속성 자체를 return한다.

- object(변수, 오프젝트) :
    - DataFrame, Series, map, Countert

## 직접 만든 모듈 불러오기

```
# my_module.py
def plus(a, b):
    return a+b
```

- main.py에서 my_module.py 속 plus 함수를 어떻게 실행할 수 있을까요?
- 모듈명.함수명을 통해 함수를 실행할 수 있습니다.

```
# main.py
import my_module
print(my_module.plus(3,8))
# 11
```

- from 모듈명 import *을 사용하면 모듈 내 함수/변수에 . 없이 접근이 가능합니다.
- 하지만 의도치 않게 다른 모듈과 함수명이 겹칠 수 있으니 각별한 주의가 필요합니다.

```
# main.py
from my_module import *

print(plus(3,8))
# 11
```

- from 모듈명 import 함수/변수명을 사용하면 .를 쓰지 않고 함수/변수를 사용할 수 있습니다.

```
# main.py
from my_module import plus
print(plus(3,8))
# 11
```

### 모듈 활용하기

```
# 모듈 불러오기
import math
import random
import pandas as pd
import numpy as np
import tensorflow as tf
…
```

- 암묵적인 네이밍 룰
    - convention
    - naming rule

- 함수의 종류
    - 원본을 변경하는 함수
        - random.shuffle() 등
        - 변경한 원본을 변수에 저장한 후 사용해야한다.
    - 원본을 변경하지 않는 함수

- 다양한 목적을 위해 만들어진 다양한 모듈들을 활용해봅시다.
    - ex) math, random, numpy, pandas, tensorflow, torch…

#### 모듈 활용하기 - math

> math – 수학 연산에 필요한 많은 함수 및 변수를 가지는 모듈

```
import math
print(math.pi) # 3.141592…
print(math.e) # 2.718281…
```

```
import math

def main():
	
    # math 모듈을 활용해보세요.
    print(math.pi) # 파이 값
    print(math.log(3)) # 로그 값
    print(math.factorial(5)) # 팩토리얼 함수
    print(math.floor(3.78)) # 내림 함수
    print(math.ceil(3.14)) # 올림 함수

if __name__ == "__main__":
    main()
```

#### 모듈 활용하기 - random

> random – 난수 생성 등 임의의 값을 생성하고 활용할 때 사용하는 모듈

```
import random

def main() :
    lst = [1,2,3,4,5,6,7,8,9]
    
    print(random.random()) # 0 < n < 1 사이 난수 생성
    # 0.48813906880058955
    print(random.randrange(1, 11)) # 1 <= n <= 10 사이의 정수 중 하나 반환
    # 3
    print(random.randint(0, 3)) # 0부터 3중 하나의 integer 값을 출력
    # 2
    random.shuffle(lst) # lst 내의 원소들의 순서를 무작위로 섞음
    print(lst)
    # [2, 9, 4, 8, 1, 3, 7, 5, 6]
    print(random.choice(lst)) # lst 내의 원소들 중 하나를 무작위로 뽑음
    # 9
    
if __name__ == "__main__":
    main()
```

- random 모듈로 로또만들기

random 모듈을 활용하여 1~45의 숫자 중 7개를 중복없이 무작위로 뽑는 함수 lotto 를 구현해보세요

```
import random

def lotto():
    # List Comprehension 사용
    lucky_numbers = [(random.randrange(1, 46)) for _ in range(7)]
    
    # 일반적인 for문 사용
    # lucky_numbers = []
    # for i in range(7) :
    #     lucky_numbers.append(random.randrange(1, 46)) for _ in range(7)
    
    # random.sample(lst, N) : lst 내의 원소들 중 N개를 샘플링
    # lucky_numbers = random.sample(range(1, 46), 7)
    
    return lucky_numbers
    
def main():
	
    lucky_numbers = lotto()
    print(lucky_numbers)
    
if __name__ == "__main__":
    main()
```

## 패키지

- 패키지(Packages)는 모듈을 폴더(Directory)로 구조화한 것입니다.
- .(dot)을 사용하여 모듈을 디렉토리 구조로 관리할 수 있게 해줍니다.

- 모듈 이름이 elice.utils인 경우,
    - elice: 패키지 이름
    - utils: elice 패키지의 utils모듈

### 패키지의 구조

패키지의 구조는 어떻게 생겼을까요?

아래와 같은 apple이라는 이름의 패키지가 존재한다고 생각해봅시다.

```
apple/__init__.py
apple/iphone/__init__.py
apple/iphone/call.py
apple/ipad/__init__.py
apple/ipad/draw.py
```

가장 상위 폴더는 패키지 이름과 동일한 디렉토리(apple)이며 다양한 서브 디렉토리(iphone, ipad)와 .py 파일들(call.py, draw.py)로 구성됩니다.

#### 패키지의 안의 함수

call.py, draw.py 파일을 아래와 같이 작성해봅니다.

패키지 안의 함수(say_hello, draw_line)를 어떻게 실행할 수 있을까요?

```
# apple/iphone/call.py
def say_hello():
    print(“Hello”)
```

```
# apple/ipad/draw.py
def draw_line():
    print(“-----------”)
```

#### 패키지의 안의 함수 실행하기

- 방법 1. 모듈을 import하여 실행할 수 있습니다.

```
# apple/iphone/call.py
import apple.iphone.call
apple.iphone.call.say_hello()
# Hello


# apple/ipad/draw.py
import apple.ipad.draw
apple.ipad.draw.draw_line():
# -----------
```

- 방법 2. 모듈이 있는 디렉토리까지 from-import하여 실행할 수 있습니다.

```
# apple/iphone/call.py
from apple.iphone import call
call.say_hello()
# Hello

# apple/ipad/draw.py
from apple.ipad import draw
draw.draw_line():
# -----------
```

- 방법 3. 모듈의 함수를 직접 import하여 실행하는 방법입니다.

```
# apple/iphone/call.py
from apple.iphone.call import say_hello
say_hello()
# Hello

# apple/ipad/draw.py
from apple.ipad.draw import draw_line
draw_line()
# -----------
```

- 쓰는 방법에 따른 import, from-import 예시

```
# 이곳에 필요한 모듈을 불러오세요.
import apple.iphone.call
from apple.ipad import draw
    
def main():
	
    apple.iphone.call.say_hello()
    draw.draw_line()
        
if __name__ == "__main__":
    main()
```


### 패키지 만들기 – init.py

> __init__.py은 무엇일까요?

- __init__.py 파일은 해당 디렉토리가 패키지의 일부임을 알려주는 역할을 합니다.

- 만약 패키지에 포함된 디렉토리에 __init__.py 파일이 없다면 패키지로 인식되지 않습니다.

- *python3.3 버전부터 __init__.py 파일이 없어도 패키지로 잘 인식하지만,
하위 버전과의 호환을 위해서 __init__.py 파일을 생성하는 것이 안전합니다.

========================

# 파일 읽고 쓰기

## 다양한 파일의 형태

- 텍스트를 담고있는 .txt
- 이미지 처리에 사용되는 .jpg, .png
- 데이터가 데이터프레임의 형태로 저장된 .csv, .xlsx

파이썬에서는 다양한 형태의 파일들을 어떻게 읽고 처리할 수 있을까요?

## 텍스트 파일 생성하기

- 내장 함수 open을 사용하여 파일을 생성할 수 있습니다.
- 파일 열기 모드로는 r-read, w-write, a-append가 있습니다.
- 파일 객체 = open(파일 이름, 파일 열기 모드)

```
f = open(“myfile.txt”, ‘w’) # 현재 디렉토리에 myfile.txt이
생성
f.write(‘I like’) # myfile.txt에 ‘I like’를 작성
f.close()
```

- 파일 열기 모드(r)를 활용하여 파일을 열 수 있습니다.
- 파일의 내용은 read(), readline(), readlines() 함수를 통해 읽을 수 있습니다.
    - read(): 파일 전체의 내용을 하나의 문자열로 읽음
    - readline() : 한 줄을 읽음
    - readlines() : 모든 줄을 읽어 리스트로 만듦

```
f = open(“myfile.txt”, ‘r’) # 현재 디렉토리에 있는 myfile.txt을 읽기
print(f.readline()) # myfile.txt의 첫 줄 ‘I like’가 출력
f.close()
```

- 파일 내용 추가 모드(a) 를 활용하여 파일에 내용을 추가할 수 있습니다.
- write 함수를 통해 이전에 쓰여진 내용 뒤에 내용을 추가할 수 있습니다.

```
f = open(“myfile.txt”, ‘a’) # 현재 디렉토리에 있는 myfile.txt을 읽기
f.write(‘ apple.’) # myfile.txt의 내용이 ‘I like apple.’로 수정
f.close()
```

### 이미지 파일 읽고 수정하기

> PIL(Python Imaging Library)을 통해 파이썬에서 이미지 파일을 읽고 수정할 수 있습니다.

```
from PIL import Image
# Image
im = Image.open(‘rabbit01.png’)
im = im.rotate(90) # 90도 회전
im.save(’rabbit02.png’) # Image 저장
```

### 이미지 파일 읽기 – matplotlib 활용

- matplotlib을 활용해서 이미지를 읽고 출력할 수도 있습니다.
- 이외에도 Python에서 이미지 파일을 읽는 다양한 방법이 존재합니다. 
    - ex) opencv, …

```
import matplotlib.pyplot as plt
a = plt.imread(‘rabbit01.png’)
plt.imshow(a)
```

### 데이터 프레임 읽기

- Pandas에 존재하는 함수(read_csv, read_excel)를 통해 csv, xlsx 파일을 데이터 프레임의 형태로 읽고 처리할 수 있습니다.

```
import pandas as pd
df = pd.read_csv(‘data.csv’, sep=‘,’) # csv 파일을 Pandas dataframe으로 읽기
df = pd.read_excel(‘data.xlsx’) # xlsx 파일을 Pandas dataframe으로 읽기
```


#### 텍스트 데이터 읽기

```
def main():
	
    # data 폴더 내의 sentences.txt 파일을 읽고 출력하세요.
    # 아래의 코드를 완성하세요.
    
    sentences = open('sentences.txt', 'r')
    
    lst = sentences.readlines()
    
    first_sentence  = lst[0]
    print(first_sentence)
    
    second_sentence = lst[1]
    print(second_sentence)
    
    last_sentence = lst[-1]
    print(last_sentence)
    
    sentences.close()

if __name__ == "__main__":
    main()
```

#### 이미지 데이터 읽기

```
from elice_utils import EliceUtils
from PIL import Image

elice_utils = EliceUtils()


def main():

    img = Image.open('rabbit01.png') # Image 읽기
    img = img.rotate(90) # 90도 회전
    img.save('rabbit_rotated.png')  # Image 저장

    elice_utils.send_image('rabbit01.png')
    elice_utils.send_image('rabbit_rotated.png')
    

if __name__ == "__main__":
    main()
```

#### csv 데이터 읽기

```
import pandas as pd


def main():
	
    # 아래의 코드를 완성하세요.
    iris = pd.read_csv('iris.csv')
    print(iris)
    print("")
    print(iris.describe())
    
if __name__ == "__main__":
    main()
```

# Jupyter Notebook 알아보기

- Jupyter Notebook은 브라우저에서 Python을 작성하고 실행할 수 있는 도구로,
데이터 분석가들에게 가장 많이 사용되는 툴입니다.

## Jupyter Notebook 시작하기

- 현재 디렉토리의 파일 목록을 볼 수 있는 화면입니다.
- New 버튼을 클릭하여 새로운 파일을 생성할 수 있습니다.
- Python 3를 클릭하면 노트북 파일(.ipynb)을 생성할 수 있습니다.
- 노트북 시작 화면으로, 제목이 Untitled.ipynb로 초기화되어 있습니다.
- 아래 셀에 파이썬 코드를 입력하고 Run버튼 클릭 or Shift+Enter로 실행합니다.
- 파이썬 코드가 실행되면 그 결과가 셀 하단 Out[ ]에 표현됩니다.
- 셀 위치는 프로그램에 영향을 미치지 않고, 셀의 실행 순서가 영향을 미칩니다.
- Markdown 형태의 셀도 생성할 수 있어서 코드 밖에서 자유로운 형태의 메모가 가능합니다.

### Jupyter Lab

- Jupyter notebook의 인터페이스를 더 발전시킨 Jupyter Lab도 있으며 한 화면에 여러 노트북을 띄울 수 있습니다.

## Google Colab 알아보기

- Google Colab은 구글에서 만든 개발환경 제공 서비스로 데이터 분석에 필요한
라이브러리가 이미 설치되어 있고 GPU 무료 액세스도 가능합니다.
- 내 서버가 필요없는 클라우드 기반의 무료 Jupyter Notebook
- 구글 계정에 로그인 한 후 구글드라이브에 접속합니다.
- 마우스 우클릭 > 더보기 > Google Colaboratory 클릭
- Colab 노트북(Untitled0.ipynb)이 생성되었습니다.
- Jupyter Notebook과 UI가 유사함을 확인할 수 있습니다.
- Jupyter Notebook과 마찬가지로 코드 셀에 코드를 입력하고 왼쪽의 플레이 버튼 클릭 or Shift+Enter로 셀을 실행할 수 있습니다.

### Google Drive와 연동하기
- 구글 드라이브 마운트를 통해 내 구글 드라이브 내 폴더 및 파일들에 자유롭게 접근할 수 있습니다.
- 좌측 탭에서 MyDrive가 마운트 된 것을 확인할 수 있습니다.

### Google Colab에서 GPU와 TPU 사용하기

- 런타임 > 런타임 유형 변경에서 GPU, TPU를 설정할 수 있습니다.
- * TPU(Tensor Processing Unit): 구글에서 발표한 데이터 분석 및 딥러닝용 하드웨어

### Google Colab 활용하기

Colab에서 다음의 라이브러리들을 import 해봅시다.
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
…
import tensorflow as tf
import torch
```

- 머신러닝, 딥러닝에 자주 활용되는 라이브러리가 이미 설치되어있어 별도의 환경설정 없이 편하게 코딩할 수 있습니다.
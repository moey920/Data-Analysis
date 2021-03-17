# Numpy(Numerical Python) : 대규모 다차원 배열을 다룰 수 있게 도와주는 라이브러리

왜 대규모 다차원 배열을 다루는가? 데이터는 대부분 숫자 배열로 볼 수 있다.(그림, 소리 등)
- 파이썬 리스트에 비해 빠르고, 메모리가 효율적이다. 

## 배열 만들기
```
list(range(10))

# [0,1,2,3,4,5,6,7,8,9]

import numpy as np

np.array([1,2,3,4,5])
# array 배열은 대괄호를 써서 표현한다.
# [1,2,3,4,5]
```

- 실수형 데이터가 들어가면, 모든 np 배열 내 데이터가 실수형으로 변환된다.
- n차원도 만들 수 있다.
- dtype='float' 등과 같이 자료형을 명시적으로 선언할 수 있다.
- array는 단일타입으로 구성된다.(list는 리스트 내 다양한 자료형을 저장할 수 있다.)
- array.dtype으로 확인할 수 있다, array.astype(int)와 같이 형변환이 가능하다.
- dtype으로는 int(i, int_, int32, int64, i8), float(f, float_, float32, float64, f8), str(str, U, U32), bool(?, bool_) 등이 있다.

### 다양한 배열 만들기
```
np.zeros(10, dtype=int)

# array([0,0,0,0,0,0,0,0,0,0])

np.ones((3,5), dtype=float)

# array([1.,1.,1.,1.,1.,],
        [1.,1.,1.,1.,1.,],
        [1.,1.,1.,1.,1.,])

np.arange(0, 20, 2) # range객체와 비슷하게 작동, 0부터 20까지 2의 간격으로 생성

# array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

np.linspace(0, 1, 5) # 0부터 1까지 5개의 구간으로 나누어 array를 만들기

# array([0., 0.25, 0.5, 0.75, 1.])
```

### 난수로 채워진 배열 만들기
```
np.random.random((2,2)) # 인자로 튜플을 받음. shape에 대한 정보이다.

np.random.normal(0, 1, (2,2)) # 정규분포도 만들 수 있다. 평균=0, 표준편차=1 이고 2x2 행렬 만들기

np.random.randint(0, 10, (2,2)) # 0부터 10까지 2x2 형태로 랜덤하게 출력

# array([난수, 난수],
        [난수, 난수])
```

# 배열의 기초
```
x2 = np.random.randint(10, size=(3, 4))
# array([[2, 2, 9, 0],
        [4, 2, 1, 0],
        [1, 8, 7, 3]])

x2.ndim # 2 # 배열이 몇 차원인지 반환
x2.shape # (3, 4) # 배열의 행렬(shape)을 반환
x2.size # 12 # 배열 내의 원소의 개수를 반환
x2.dtype # dtype(‘int64’) # 배열의 데이터 타입을 반환
```

## 찾고 잘라내기 (Indexing/Slicing)

> Indexing: 인덱스로 값을 찾아냄(리스트와 비슷하다)

```
x = np.arange(7)
# 0 1 2 3 4 5 6
x[3]
# 3
x[7]
# IndexError: index 7 is out of bounds
x[0] = 10 # 인덱스로 값을 찾아 값 변경
# array([10, 1, 2, 3, 4, 5, 6])
```

> Slicing: 인덱스 값으로 배열의 부분을 가져옴

```
x = np.arange(7)
# 0 1 2 3 4 5 6
x[1:4] # start ~ end-1 까지
# array([1, 2, 3])
x[1:]
# array([1, 2, 3, 4, 5, 6])
x[:4] # end-1 이전의 모든 원소
# array([0, 1, 2, 3])
x[::2] # step을 이용, 전체 배열에서 2씩 건너뛰면서 반환
array([0, 2, 4, 6])
```

```
import numpy as np

print("1차원 array")
array = np.arange(10)
print(array)

# Q1. array의 자료형을 출력해보세요. 변수의 자료형을 구할 수 있다. array라는 변수가 파이썬에서 어떤 자료형으로 저장되고 있는지 확인
print(type(array))

# Q2. array의 차원을 출력해보세요.
print(array.ndim)

# Q3. array의 모양을 출력해보세요.
print(array.shape)

# Q4. array의 크기를 출력해보세요.
print(array.size)

# Q5. array의 dtype(data type)을 출력해보세요.
print(array.dtype)

# Q6. array의 인덱스 5의 요소를 출력해보세요.
print(array[5])

# Q7. array의 인덱스 3의 요소부터 인덱스 5 요소까지 출력해보세요.
print(array[3:6])
```

```
import numpy as np

print("2차원 array")
matrix = np.arange(1, 16).reshape(3,5)  #1부터 15까지 들어있는 (3,5)짜리 배열을 만듭니다.
print(matrix)


# Q1. matrix의 자료형을 출력해보세요.
print(type(matrix))

# Q2. matrix의 차원을 출력해보세요.
print(matrix.ndim)

# Q3. matrix의 모양을 출력해보세요.
print(matrix.shape)

# Q4. matrix의 크기를 출력해보세요.
print(matrix.size)

# Q5. matrix의 dtype(data type)을 출력해보세요.
print(matrix.dtype)

# Q6. matrix의 (2,3) 인덱스의 요소를 출력해보세요.
print(matrix[2,3])

# Q7. matrix의 행은 인덱스 0부터 인덱스 1까지, 열은 인덱스 1부터 인덱스 3까지 출력해보세요.
print(matrix[0:2, 1:4])
```

## Reshape & 이어 붙이고 나누기

### 모양 바꾸기
- reshape: array의 shape를 변경

```
x = np.arange(8)
x.shape
# (8,) : 1차원 데이터 배열(1x8)
x2 = x.reshape((2, 4))
# array([[0, 1, 2, 3],
        [4, 5, 6, 7]])
x2.shape
# (2, 4) # 2차원 데이터 배열(2x4)
```

### 이어 붙이고 나누기

- concatenate: array를 이어 붙임

```
x = np.array([0, 1, 2])
y = np.array([3, 4, 5])
np.concatenate([x, y])
# array([0, 1, 2, 3, 4, 5])
```

- np.concatenate: axis 축을 기준으로 이어붙임 (axis = 0(세로),1(가로) 기억하기)

```
matrix = np.arange(4).reshape(2, 2)
# 0 1
# 2 3

np.concatenate([matrix, matrix], axis=0) # axis=0, 아래 방향으로 붙게 된다.(4x2)
# 0 1
# 2 3
# 0 1
# 2 3

np.concatenate([matrix, matrix], axis=1) # (2x4)
# 0 1 0 1
# 2 3 2 3
```

- np.split: axis 축을 기준으로 분할

```
matrix = np.arange(16).reshape(4, 4)
upper, lower = np.split(matrix, [3], axis=0) # 나눌 array, 어디 인덱스를 나눌 것인지(3번쨰 인덱스를 기준으로, 세로로 나누어준다)
# 0 1 2 3
# 4 5 6 7
# 8 9 10 11
# 12 13 14 15

upper = 
# 0 1 2 3
# 4 5 6 7
# 8 9 10 11
lower = 
# 12 13 14 15

left, right = np.split(matrix, [3], axis=1)

left = 
# 0 1 2
# 4 5 6
# 8 9 10
# 12 13 14
right = 
# 3
# 7
# 11
# 15
```

```
import numpy as np

print("array")
array = np.arange(8)
print(array)
print("shape : ", array.shape, "\n")

# Q1. array를 (2,4) 크기로 reshape하여 matrix에 저장한 뒤 matrix와 그의 shape를 출력해보세요.
print("# reshape (2, 4)")
matrix = array.reshape(2,4)


print(matrix)
print("shape : ", matrix.shape)
```

```
import numpy as np

print("matrix")
matrix = np.array([[0,1,2,3],
                   [4,5,6,7]])
print(matrix)
print("shape : ", matrix.shape, "\n")

# (아래의 배열 모양을 참고하세요.)
# Q1. matrix 두 개를 세로로 붙이기 
'''
[[0 1 2 3]
 [4 5 6 7]
 [0 1 2 3]
 [4 5 6 7]]
'''
m = np.concatenate([matrix, matrix], axis=0)
print(m)


# Q2. matrix 두 개를 가로로 붙이기
'''
[[0 1 2 3 0 1 2 3]
 [4 5 6 7 4 5 6 7]]
'''
n = np.concatenate([matrix, matrix], axis=1)
print(n)
```

```
import numpy as np

print("matrix")
matrix = np.array([[ 0, 1, 2, 3],
                   [ 4, 5, 6, 7],
                   [ 8, 9,10,11], 
                   [12,13,14,15]])
print(matrix, "\n")

# Q1. matrix를 [3] 행에서 axis 0으로 나누기
a, b = np.split(matrix, [3], axis=0)

print(a, "\n")
print(b, "\n")


# Q2. matrix를 [1] 열에서 axis 1로 나누기
c, d = np.split(matrix, [1], axis=1)

print(c, "\n")
print(d)
```

# NumPy 연산

> 루프는 느리다

- array의 모든 원소에 5를 더해서 만드는 함수
    - 만약 array의 크기가 크다면..?

```
def add_five_to_array(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = values[i] + 5
    return output

values = np.random.randint(1, 10, size=5)
add_five_to_array(values)

big_array = np.random.randint(1, 100, size=10000000)
add_five_to_array(big_array)
# 5.3 s ± 286 ms per loop (mean ± std. dev. of 7 runs,5 loops each)

big_array + 5
# 33.5 ms ± 1.94 ms per loop (mean ± std. dev. of 7runs, 5 loops each)
```

## array는 +, - *, / 에 대한 기본 연산을 지원

```
x = np.arange(4)
# array([0, 1, 2, 3])
x + 5
# array([5, 6, 7, 8])
x - 5
# array([-5, -4, -3, -2])
x * 5
# array([ 0, 5, 10, 15])
x / 5
# array([0. , 0.2, 0.4, 0.6])
```

## 행렬간 연산 : 다차원 행렬에서도 적용 가능

```
x = np.arange(4).reshape((2, 2))
y = np.random.randint(10, size=(2, 2))

x + y
# array([[1, 7],
        [6, 5]])

x - y
# array([[-1, -5],
        [-2, 1]])
```

```
import numpy as np

array = np.array([1,2,3,4,5])
print(array)


# Q1. array에 5를 더한 값을 출력해보세요.
print(array + 5)

# Q2. array에 5를 뺀 값을 출력해보세요.
print(array - 5)

# Q3. array에 5를 곱한 값을 출력해보세요.
print(array * 5)

# Q4. array를 5로 나눈 값을 출력해보세요.
print(array / 5)


# Q5. array에 array2를 더한 값을 출력해보세요.    
array2 = np.array([5,4,3,2,1])
print(array + array2)


# Q6. array에 array2를 뺀 값을 출력해보세요.
print(array - array2)
```

# 브로드캐스팅(Broadcasting) : shape이 다른 array끼리 연산
```
matrix = 
2 4 2
6 5 9
9 4 7

matrix + 5 = 
7 9 7
11 10 14
14 9 12
```

```
matrix = 
2 4 2
6 5 9
9 4 7
np.array([1, 2, 3]) = 
1 2 3 

matrix + np.array([1, 2, 3]) = 
3 6 5
7 7 12
10 6 10
```

```
np.arange(3).reshape((3,1)) = 
0
1
2
np.arange(3) = 
0 1 2 

np.arange(3).reshape((3,1)) + np.arange(3) =
0 1 2
1 2 3
2 3 4
```

```
import numpy as np

'''
[[0]
 [1]
 [2]
 [3]
 [4]
 [5]] 배열 A와

 [0 1 2 3 4 5] 배열 B를 선언하고, 덧셈 연산해보세요.
'''

A = np.arange(6).reshape(6,1)
B = np.arange(6)

print(A+B)
[[ 0  1  2  3  4  5]
 [ 1  2  3  4  5  6]
 [ 2  3  4  5  6  7]
 [ 3  4  5  6  7  8]
 [ 4  5  6  7  8  9]
 [ 5  6  7  8  9 10]]
```

# 집계함수 & 마스킹 연산

## 집계함수

> 집계: 데이터에 대한 요약 통계

```
x = np.arange(8).reshape((2, 4))
np.sum(x) # 모든 원소의 합
# 28
np.min(x) # 최소값
# 0
np.max(x) # 최대값
# 7
np.mean(x) # 평균값
# 3.5
np.std(x) # 표준편차
```

```
x = np.arange(8).reshape((2, 4))
[[0, 1, 2, 3],
 [4, 5, 6, 7]]

np.sum(x, axis=0) # 세로축을 더해주세요
# array([ 4, 6, 8, 10])

np.sum(x, axis=1)
# array([ 6, 22])
```

## 마스킹 연산: True, False array를 통해서 특정 값들을 뽑아내는 방법

```
x = np.arange(5)
# array([0, 1, 2, 3, 4])
x < 3
# array([ True, True, True, False, False])
x > 5
# array([False, False, False, False, False])
x[x < 3]
# array([0, 1, 2])
```

```
import numpy as np

matrix = np.arange(8).reshape((2, 4))
print(matrix)

# Q1. sum 함수로 matrix의 총 합계를 구해 출력해보세요.
print(np.sum(matrix))

# Q2. max 함수로 matrix 중 최댓값을 구해 출력해보세요.
print(np.max(matrix))

# Q3. min 함수로 matrix 중 최솟값을 구해 출력해보세요.
print(np.min(matrix))

# Q4. mean 함수로 matrix의 평균값을 구해 출력해보세요.
print(np.mean(matrix))

# Q5. sum 함수의 axis 매개변수로 각 열의 합을 구해 출력해보세요.
print(np.sum(matrix, axis=0))

# Q6. sum 함수의 axis 매개변수로 각 행의 합을 구해 출력해보세요.
print(np.sum(matrix, axis=1))

# Q7. std 함수로 matrix의 표준편차를 구해 출력해보세요.
print(np.std(matrix))

# Q8. 마스킹 연산을 이용하여 matrix 중 5보다 작은 수들만 추출하여 출력해보세요.
print(matrix[matrix < 5])
```

```
import numpy as np

daily_liar_data = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]


# 양치기 소년이 거짓말을 몇 번 했는지 구하여 출력해주세요.
liar_array = np.array(daily_liar_data)
print(len(liar_array[liar_array == 0]))
```

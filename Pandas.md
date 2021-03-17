# Pandas
- 구조화된 데이터를 효과적으로 처리하고 저장할 수 있는 라이브러리
- Array 계산에 특화된 numpy를 기반으로 만들어져서 다양한 기능 제공
- 엑셀과 같은 스프레드시트나 DB 사용자들에게 익숙한 경험을 제공한다.

## Pandas Series

> Series : 특수한 dictionary, numpy array가 보강된 형태로써 Data와 Index를 가지고있다.

- 인덱스를 가지고 있고 인덱스로 접근 가능
```
data = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
a 1
b 2
c 3
d 4

data['b’]
# 2
```

- 딕셔너리를 가지고 Series 데이터를 만들 수 있다.
```
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}

population = pd.Series(population_dict)

china 141500
japan 12718
korea 5180
usa 32676
dtype: int64

population.values
=> numpy.array로 나온다.(numpy 객체로 이루어져있다.)
```

```
import numpy as np
import pandas as pd

# 예시) 시리즈 데이터를 만드는 방법.
series = pd.Series([1,2,3,4], index = ['a', 'b', 'c', 'd'], name="Title")
print(series, "\n")


# 국가별 인구 수 시리즈 데이터를 딕셔너리를 사용하여 만들어보세요.
country = {
    'korea' : 5180,
    'japan' : 12718,
    'china' : 141500,
    'usa' : 32676
}
print(country)

country = pd.Series(country, name="country")
print(country)
```

# DataFrame
- 여러 개의 Series가 모여서 행과 열을 이룬 데이터

```
gdp_dict = {
'korea': 169320000,
'japan': 516700000,
'china': 1409250000,
'usa': 2041280000,
}

country = {
    'korea' : 5180,
    'japan' : 12718,
    'china' : 141500,
    'usa' : 32676
}

gdp = pd.Series(gdp_dict)
population = pd.Series(country)

country = pd.DataFrame({
    'population': population,
    'gdp': gdp
})

country.index
# Index(['china', 'japan', 'korea', 'usa'], dtype='object’)

country.columns
# Index(['gdp', 'population'], dtype='object’) # python object(객체), df에서 문자열은 객체로 본다.

country['gdp’] # Series 데이터 출력(numpy가 보강된 형태)
china 1409250000
japan 516700000
korea 169320000
usa 2041280000
Name: gdp, dtype: int64

type(country['gdp’])
# pandas.core.series.Series
```

- Series도 numpy array처럼 연산자를 활용

```
# 1인당 gdp = gdp/인구수
gdp_per_capita = country['gdp'] / country['population’] # Series형태로 나옴(index는 그대로 유지)
country['gdp per capita’] = gdp_per_capita # df에 새로운 Column으로 넣어준다.
```

## 저장과 불러오기
- 만든 데이터 프레임을 저장할 수 있다
```
country.to_csv(“./country.csv”) # ,로 구분된 값
country.to_excel(“country.xlsx”)

country = pd.read_csv(“./country.csv”)
country = pd.read_excel(“country.xlsx”)
```

```
import numpy as np
import pandas as pd

# 두 개의 시리즈 데이터가 있습니다.
print("Population series data:")
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}
population = pd.Series(population_dict)
print(population, "\n")

print("GDP series data:")
gdp_dict = {
    'korea': 169320000,
    'japan': 516700000,
    'china': 1409250000,
    'usa': 2041280000,
}
gdp = pd.Series(gdp_dict)
print(gdp, "\n")


# 이곳에서 2개의 시리즈 값이 들어간 데이터프레임을 생성합니다.
print("Country DataFrame")
country = pd.DataFrame({
    'population' : population_dict,
    'gdp' : gdp_dict
})
# print(country)


# 데이터 프레임에 gdp per capita 칼럼을 추가하고 출력합니다.
gdp_per_capita = country['gdp'] / country['population'] 
country['gdp per capita'] = gdp_per_capita
print(country)


# 데이터 프레임을 만들었다면, index와 columns도 각각 확인해보세요.
print(country.index)

print(country.columns)
```


# Indexing/Slicing

- .loc : **명시적인 인덱스를 참조**하는 인덱싱/슬라이싱

```
country.loc['china']
country.loc['japan':'korea', :'population'] # 슬라이싱도 가능, 인덱스는 j~k까지, 컬럼은 0~population까지
```

- .iloc : **파이썬 스타일 정수 인덱스** 인덱싱/슬라이싱

```
country.iloc[0] # index : China
country.iloc[1:3, :2] # index : japan, korea / columns : gdp, population
```

## DataFrame 새 데이터 추가/수정

- 1. 리스트로 추가
- 2. 딕셔너리로 추가

```
dataframe = pd.DataFrame(columns=['이름','나이','주소'])
dataframe.loc[0] = ['임원균', '26','서울’] # 리스트로 추가하기
dataframe.loc[1] = {'이름':'철수','나이':'25','주소':'인천’} # 딕셔너리로 추가하기
dataframe.loc[1,'이름'] = '영희’ # df 수정하기, index = 1, column = '이름'
```

- DataFrame 새로운 컬럼 추가

```
dataframe[‘전화번호'] = np.nan # 빈 값으로 추가, nan = not a number(숫자가 아닌, 값이 빈 데이터)
dataframe.loc[0,‘전화번호’] = ‘01012341234’
len(dataframe)
# 2
```

### 컬럼 선택하기
- 컬럼 이름이 하나만 있다면 Series
- 컬럼이 리스트로 들어가 있다면 DataFrame
```
dataframe["이름"] # Series
dataframe[["이름","주소","나이"]] # DataFrame, 입력한 컬럼한 입력한 순으로 출력된다.
```

```
import numpy as np
import pandas as pd

# 첫번째 컬럼을 인덱스로 country.csv 파일 읽어오기.
print("Country DataFrame")
country = pd.read_csv("./data/country.csv", index_col=0) # df를 만들 땐 csv파일을 이용해서 만들 수도 있다.
print(country, "\n")

# 명시적 인덱싱을 사용하여 데이터프레임의 "china" 인덱스를 출력해봅시다.
print(country.loc['china'])


# 정수 인덱싱을 사용하여 데이터프레임의 1번째부터 3번째 인덱스를 출력해봅시다.
print(country.iloc[1:4])
```


# Pandas연산과 함수

- 튜토리얼의 데이터와 다르게 현실 데이터는 일부 누락되어 있는 형태가 많음

## 누락된 데이터 체크
```
dataframe.isnull() # 데이터가 비어있는지 체크(비어있으면 True) : nan, None 체크
dataframe.notnull() # 데이터가 비어있지 않은 경우 체크(비어있지 않으면 True)

dataframe.dropna() # 비어있는 데이터를 삭제한다(row 전체를 삭제)
dataframe['전화번호'] = dataframe['전화번호'].fillna('전화번호 없음') # 해당 colum에 비어있는 데이터를 fillna의 인자로 채운다.
```
### Series연산
- numpy array에서 사용했던 Series연산자들을 동일하게 활용할 수 있음
```
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])

A + B
0 NaN # B엔 인덱스 0이 없어서 값을 알 수 없음
1 5.0
2 9.0
3 NaN # A엔 인덱스 3이 없어서 값을 알 수 없음
dtype: float64

A.add(B, fill_value=0) # 없는 인덱스를 생성하고 value를 0으로 채워서 계산한다.
0 2.0
1 5.0
2 9.0
3 5.0
dtype: float64
```

### DataFrame 연산

- add ( + ), sub ( - ), mul ( * ), div ( / )
```
A = pd.DataFrame(np.random.randint(0, 10, (2, 2)), columns=list("AB"))
B = pd.DataFrame(np.random.randint(0, 10, (3, 3)), columns=list("BAC")) # AB와 컬럼 순서가 다르니 맞춰서 계산된다.

A + B
    A   B   C
0 4.0   4.0 NaN
1 4.0   1.0 NaN
2 NaN   NaN NaN

A.add(B, fill_value=0)
    A   B   C
0 4.0   4.0 3.0
1 4.0   1.0 1.0
2 0.0   6.0 4.0
```

#### 집계함수
- numpy array에서 사용했던 sum, mean 등의 집계함수를 동일하게 사용할 수 있음

```
data = {
'A': [ i+5 for i in range(3) ], # [5,6,7]
'B': [ i**2 for i in range(3) ] # [0,1,4]
}
df = pd.DataFrame(data) # df만들기

df['A'].sum() # 18 # Series A 데이터만 뽑아서 더하기
df.sum() # A(index) : 18(value), B : 5
df.mean() # A(index) : 6(value), B : 1.666
```

```
import numpy as np
import pandas as pd


print("A: ")
A = pd.DataFrame(np.random.randint(0, 10, (2, 2)), columns=['A', 'B'])      #칼럼이 A, B입니다.
print(A, "\n")


print("B: ")
B = pd.DataFrame(np.random.randint(0, 10, (3, 3)), columns=['B', 'A', 'C'])     #칼럼이 B, A, C입니다.
print(B, "\n")


# 아래에 다양한 연산을 자유롭게 적용해보세요.
print(A+B)
print(A.add(B, fill_value=0))

print(A-B)
print(A.sub(B, fill_value=0))

print(A*B)
print(A.mul(B, fill_value=0))

print(A/B)
print(A.div(B, fill_value=0))
```

## Dataframe 정렬하기

### 값으로 정렬하기 : sort_values()

```
df = pd.DataFrame({
'col1' : [2, 1, 9, 8, 7, 4],
'col2' : ['A', 'A', 'B', np.nan, 'D', 'C'],
'col3': [0, 1, 9, 4, 2, 3],
})

df.sort_values('col1') # 해당 컬럼의 값을 기준으로 sorting 된다.

df.sort_values('col1', ascending=False) # 내림차순으로 정렬한다.

df.sort_values(['col2', 'col1']) # col2를 먼저 정렬하고, col2 내의 같은 값에 대해서는 col1을 정렬한다.
```

```
import numpy as np
import pandas as pd

print("DataFrame: ")
df = pd.DataFrame({
    'col1' : [2, 1, 9, 8, 7, 4],
    'col2' : ['A', 'A', 'B', np.nan, 'D', 'C'],
    'col3': [0, 1, 9, 4, 2, 3],
})
print(df, "\n")


# 정렬 코드 입력해보기    
# Q1. col1을 기준으로 오름차순으로 정렬하기.
sorted_df1 = (df.sort_values('col1'))


# Q2. col2를 기준으로 내림차순으로 정렬하기.
sorted_df2 = (df.sort_values('col2', ascending = False))


# Q3. col2를 기준으로 오름차순으로, col1를 기준으로 내림차순으로 정렬하기.(복수의 정렬기준 사용방법)
sorted_df3 = (df.sort_values(['col2', 'col1'], ascending = [True, False]))
```

```
import pandas as pd

tree_df = pd.read_csv('./data/tree_data.csv')

# ./data/tree_data.csv 파일을 읽어서 작업해보세요!
# print(len(tree_df.dropna())) # nan값이 있는지 확인하고 제거한 후 개수 확인

tree_df = tree_df.sort_values('height', ascending=False)
# print(tree_df.iloc[:5])
# print(tree_df.head(5)) # iloc[:5] 와 같다
print(tree_df.iloc[0])
tree_df.to_csv("./data/tree_df.csv")
```
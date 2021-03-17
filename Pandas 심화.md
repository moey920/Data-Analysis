# Pandas 활용하기

## 조건으로 검색하기 (masking 연산)
- numpy array와 마찬가지로 masking 연산이 가능하다.

```
import numpy as np
import pandas as pd
df = pd.DataFrame(np.random.rand(5, 2), columns=["A", "B"])
df[“A”] < 0.5
```
```
결과 : 
0 True
1 True
2 True
3 False
4 False
Name: A, dtype: bool
```

- 조건에 맞는 DataFrame row를 추출 가능하다
```
import numpy as np
import pandas as pd
df = pd.DataFrame(np.random.rand(5, 2), columns=["A", "B"])
df[(df["A"] < 0.5) & (df["B"] > 0.3)] # &는 둘다 True인 경우를 반환, df를 두번 감싸 값을 얻는다.
df.query("A < 0.5 and B > 0.3") # df.query() 함수를 이용해서 값을 반환할 수도 있다.
```

- 문자열이라면 다른 방식으로도 조건 검색이 가능하다
```
df["Animal"].str.contains("Cat") # df의 Animal 컬럼내의 문자열에서 "Cat"이라는 문자열을 포함하면 True
df.Animal.str.match("Cat") # 정규표현식
df["Animal"] == "Cat"
```

## 함수로 데이터 처리하기 : apply()
- apply를 통해서 함수로 데이터를 다룰 수 있다

```
df = pd.DataFrame(np.arange(5), columns=["Num"])
def square(x):
    return x**2
df["Num"].apply(square) # 함수 자체를 인자로 넣는다, 결과는 Series
df["Square"] = df.Num.apply(lambda x: x ** 2) # 새로운 column생성, Num에 함수를 적용시켜 생성한다.
```

```
df = pd.DataFrame(columns=["phone"])
df.loc[0] = "010-1234-1235"
df.loc[1] = "공일공-일이삼사-1235"
df.loc[2] = "010.1234.일이삼오"
df.loc[3] = "공1공-1234.1이3오"
df["preprocess_phone"] = ''

def get_preprocess_phone(phone):
    mapping_dict = {
        "공": "0",
        "일": "1",
        "이": "2",
        "삼": "3",
        "사": "4",
        "오": "5",
        "-": "",
        ".": "",
    }
    for key, value in mapping_dict.items():
        phone = phone.replace(key, value)
    return phone

df["preprocess_phone"] = df["phone"].apply(get_preprocessed_phone)
```

- replace: apply 기능에서 데이터 값만 대체 하고 싶을때

```
df.Sex.replace({"Male": 0, "Female": 1}) # Male, Female로 이루어진 데이터를 0과 1로 대치함
df.Sex.replace({"Male": 0, "Female": 1}, inplace=True) # Serise 데이터를 반환하는 것이라니라, df 내 데이터를 한번에 바꿔준다.
```

## 그룹으로 묶기

- 간단한 집계가 아닌 조건부로 집계를 하고 싶은 경우

```
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],'data1': [1,2,3,4,5,6], 'data2': np.random.randint(0,6,6)})
df.groupby('key’)
# <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x10e3588> # df의 GroupBy 객체이다.
df.groupby('key').sum() # data1, data2를 각각 더한다.
df.groupby(['key','data1']).sum()
```

- aggregate : groupby를 통해서 **집계를 한번에 계산**하는 방법
```
df.groupby('key').aggregate(['min', np.median, max])
df.groupby('key').aggregate({'data1': 'min', 'data2': np.sum}) # 컬럼마다 다른 연산을 적용해서 집계할 수 있다.
```

- filter : groupby를 통해서 그룹 속성을 기준으로 **데이터 필터링**
```
def filter_by_mean(x):
    return x['data2'].mean() > 3
df.groupby('key').mean() # A 그룹은 평균이 3보다 작기때문에 
df.groupby('key').filter(filter_by_mean) # 평균이 3보다 큰 데이터만 필터링 된다.
```
- apply :groupby를 통해서 묶인 데이터에 함수 적용
```
df.groupby('key').apply(lambda x: x.max() - x.min()) # 그룹화할 때 각 키의 최대값에서 최소값을 빼서 반환한다.
```

- get_group : groupby로 묶인 데이터에서 key값으로 데이터를 가져올 수 있다
```
df = pd.read_csv("./univ.csv")
df.head()
df.groupby("시도").get_group("충남") # groupby로 시도별로 묶고(충남,경기...) 그 중 충남의 데이터만 반환
len(df.groupby("시도").get_group("충남"))
```

```
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'key': ['A', 'B', 'C', 'A', 'B', 'C'],
    'data1': [0, 1, 2, 3, 4, 5],
    'data2': [4, 4, 6, 0, 6, 1]
})
print("DataFrame:")
print(df, "\n")

# key를 기준으로 묶어 합계를 구해 출력해보세요.
print(df.groupby('key').sum())


# key와 data1을 기준으로 묶어 합계를 구해 출력해보세요.
print(df.groupby(['key', 'data1']).sum())

# aggregate를 이용하여 요약 통계량을 산출해봅시다.
# 데이터 프레임을 'key' 칼럼으로 묶고, data1과 data2 각각의 최솟값, 중앙값, 최댓값을 출력하세요.
print(df.groupby('key').aggregate([min, np.median, max]))

# 데이터 프레임을 'key' 칼럼으로 묶고, data1의 최솟값, data2의 합계를 출력하세요.
print(df.groupby('key').aggregate({'data1' : min, 'data2' : np.sum}))

# groupby()로 묶은 데이터에 filter를 적용해봅시다.
# key별 data2의 평균이 3이 넘는 인덱스만 출력해봅시다.
print("filtering : ")
def filter_by_mean(x) :
    return x['data2'].mean() > 3
print(df.groupby('key').filter(filter_by_mean))

# groupby()로 묶은 데이터에 apply도 적용해봅시다.
# 람다식을 이용해 최댓값에서 최솟값을 뺀 값을 적용해봅시다.
print("applying : ")
print(df.groupby('key').apply(lambda x : x.max() - x.min()))
```

## MultiIndex & pivot_table
- 인덱스를 계층적으로 만들 수 있다
- MultiIndex
```
df = pd.DataFrame(
    np.random.randn(4, 2),
    index=[['A', 'A', 'B', 'B'], [1, 2, 1, 2]], # 행 인덱스가 2차원 리스트
    columns=['data1', 'data2']
)
```

- 열 인덱스도 계층적으로 만들 수 있다
```
df = pd.DataFrame(
np.random.randn(4, 4),
columns=[["A", "A", "B", "B"], ["1", "2", "1", "2"]] # 열 인덱스가 2차원 리스트
)
```

- 다중 인덱스 컬럼의 경우 인덱싱은 계층적으로 한다
- 인덱스 탐색의 경우에는 loc, iloc를 사용가능하다
```
df[“A”] # A계층 아래의 전체의 열을 반환
df[“A”][“1”] # A 계층 아래의 1열만 반환
```

- pivot_table
    - 데이터에서 필요한 자료만 뽑아서 새롭게 요약,
    - 분석 할 수 있는 기능 엑셀에서의 피봇 테이블과 같다
    - Index : 행 인덱스로 들어갈 **key**
    - Column : 열 인덱스로 **라벨링될 값**
    - Value : 분석할 **데이터**

- 타이타닉 데이터에서 성별과 좌석별 생존률 구하기
```
df.pivot_table(
    index='sex', columns='class', values='survived’, # 성별을 key로 좌석별 생존율
    aggfunc=np.mean # 어떻게 값을 채울것인지 정하는게 aggfunc. 추출한 데이터의 평균으로 채운다.
)
```

```
pivot_table
df.pivot_table(
    index="월별", columns='내역', values=["수입", '지출'])
```

- 피리 부는 사나이를 따라가는 아이들

```
import pandas as pd
import numpy as np

# 경로: "./data/the_pied_piper_of_hamelin.csv"
df = pd.read_csv("./data/the_pied_piper_of_hamelin.csv")
# print(df)

# 피리부는 사나이 데이터에서 아이들만 골라내는 데 마스킹 연산
children = df[df["구분"] == "Child"]
print(children)

# 피리부는 사나이를 따라간 아이들의 일차별 연령을 계산하는 데 groupby 함수
mean_age = children.groupby("일차").mean()
print(mean_age)

# 아이들의 일차별 연령을 성별로 나누어 표로 출력하는 데 pivot table을 이용
sex_mean_age = children.pivot_table(index="일차", columns="성별", values="나이", aggfunc=np.mean)
print(sex_mean_age)

# 피리부는 사나이를 따라간 아이들의 이름을 중복없이 출력
for name in children["이름"].unique() :
    print(name)
```
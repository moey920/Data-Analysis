
# coding: utf-8

# # [실습] 프로젝트 (2) : Tip 데이터 분석하기

# ### Tip 데이터셋에서 가장 높은 Tip을 받기 위한 전략 짜기

# ### 학습 목표
# - 각 테이블 별로 전체 금액, 팁 금액, 성별 등의 정보가 담겨있는 Tip 데이터셋을 자세히 살펴본다.
# - 데이터를 pandas, numpy, matplotlib 등의 패키지로 다루는 데에 익숙해진다.
# - 각 Column별로 Tip이 높아지는 경향성을 찾아보며 어떤 특징의 테이블에 언제 가야 더 많이 받을 수 있는지 분석해본다.

# ---

# ## Contents

# #### 1. [데이터 분석 준비하기]()
# 
# #### 2. [데이터셋 뜯어보기]()
# 
# #### 3. [가설을 세우고 그에 따라 세부 분석해보기]()
# - 가설 : 테이블의 인원수를 나타내는 `size` 컬럼은 tip을 결정짓는 중요한 요소일 것이다

# ---

# # 1. 데이터 분석 준비하기

# ### Import Modules

# 데이터 분석에 필요한 module을 import 해 봅니다.    
# 지금까지 배웠던 numpy, pandas, matplotlib과 함께 또다른 시각화 툴인 seaborn 까지 사용해 보겠습니다.

# In[1]:


# numpy
import numpy as np

# pandas
import pandas as pd

# seaborn
import seaborn as sns

#matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load Dataset

# 이번 데이터셋은 데이터 시각화 패키지인 seaborn 패키지 안에 포함되어 있습니다.     
# 
# - seaborn 공식 홈페이지 : http://seaborn.pydata.org/
# 
# 
# seaborn은 간단한 데이터셋으로 여러 실험을 해볼 수 있도록 샘플 데이터셋을 제공합니다.    
# 제공되는 데이터셋은 다음 함수로 확인할 수 있습니다.

# In[3]:


# 빨간 warning message를 보고싶지 않다면
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# seaborn 데이터 살펴보기
sns.get_dataset_names()


# 비행 데이터를 담은 `flights`, 붓꽃 데이터를 담은 `iris` 등 여러가지 데이터가 있네요.    
# 이 중 우리가 오늘 사용해 볼 데이터셋은 Tip 데이터입니다. `sns.load_dataset("tips")` 으로 데이터셋을 가져와 보겠습니다.    

# In[8]:


# 데이터 불러오기
# pd.read_csv('tips.csv')
data = sns.load_dataset("tips")
tips = data
print(tips.shape)
tips.head()


# 총 244개의 행과 7개의 열로 이루어져 있군요.     
# 
# 각 컬럼에 대한 세부 내용은 다음과 같습니다.    
# - `total_bill` : 테이블의 전체 지불 금액
# - `tip` : 팁으로 낸 금액
# - `sex` : 웨이터의 성별
# - `smoker` : 손님의 흡연 여부
# - `day` : 요일
# - `time`  : 시간
# - `size` : 테이블 인원수

# In[7]:


# info() 로 데이터 한 눈에 확인하기
tips.info()


# 이번에도 결측치 먼저 확인해보겠습니다.

# In[10]:


# 결측치 개수 확인하기
# True의 개수 구하기(Null 값의 합)
tips.isnull().sum()


# 다행히 결측치는 아무것도 없군요! 그렇다면 바로 진행해보도록 하겠습니다.

# ### `tip_rate` 컬럼 추가
# `tip`을 `total_bill`로 나눈 `tip_rate` 컬럼을 새로 만들어보겠습니다.

# In[13]:


# 팁 비율 컬럼 추가하기
# 소수점을 반올림하기 : round(반올림 하고싶은 인자, 소수점 아래 어디까지 표시할 것인지 표시)
tips['tip_rate'] = round((tips['tip'] / tips['total_bill']) * 100, 2)
tips.head()


# ---

# # 2. 데이터셋 뜯어보기
# 본격적으로 데이터셋을 하나하나 뜯어보도록 합니다. 데이터를 완벽하게 이해해야 원하는 결론도 낼 수 있다는 점을 잊지 마세요!

# ### (1) `sex` : 웨이터의 성별

# 성별은 어떤 데이터로 이루어져 있고, 각 데이터는 몇 개씩 있을까요?    
# 
# 
# 
# 먼저 pandas의 `groupby` 함수를 이용해서 확인해보겠습니다.   

# In[15]:


# 웨이터의 성별 값 확인하기
# 값을 보고자한다면 정확하게 어떤 값을 원하는지 명시해주어야한다.
# size()는 빈도를 반환하기 때문에 int 자료형을 갖는다.
tips.groupby('sex').size()


# 위 결과는 **Series**의 데이터 자료형으로 표현된 것입니다.
# 
# 위 형태를 pandas의 DataFrame으로 보고싶다면 `reset_index`를 활용해서 index를 만들어 주면 됩니다.

# In[17]:


# 성별을 pandas Dataframe 형식으로 보기
sex_count = tips.groupby('sex').size().reset_index(name = "counts")
sex_count


# 다음은 그래프로도 확인해 보겠습니다.     

# In[18]:


# 그래프로 성별 확인해보기
sns.countplot(data = tips, x = "sex")
plt.show()


# 정확한 비율이 궁금하다면 다음과 같이 계산해볼 수 있죠!

# In[19]:


print("Male   ", "{:.1f}%".format(sex_count.loc[0, "counts"] / len(tips) * 100))
print("Female ", "{:.1f}%".format(sex_count.loc[1, "counts"] / len(tips) * 100))


# ### (2) `smoker` : 손님의 흡연 여부

# smoker의 데이터 개수는 이제 간단하게 확인할 수 있겠습니다.

# In[21]:


# groupby 를 활용해 흡연 여부 데이터 프레임 형식으로 보기
smoker_count = tips.groupby('smoker').size().reset_index(name = "counts")
smoker_count


# In[22]:


# 데이터 시각화해보기 
sns.countplot(data = tips, x = "smoker")
plt.show()


# In[24]:


print("smoker   ", "{:.1f}%".format(smoker_count.loc[0, "counts"] / len(tips) * 100))
print("non-smoker ", "{:.1f}%".format(smoker_count.loc[1, "counts"] / len(tips) * 100))


# ### (3) `day`, `time` : 요일, 시간

# 이번에는 day와 time 별로 데이터 상황을 살펴보겠습니다.    
# 
# 
# 여러 값이 있는 데이터를 살펴보는 것은 `pivot_table`를 활용하면 간단합니다.     
# `pivot_table`에서 데이터의 개수를 세기 위해 `counts`라는 컬럼을 만들어주겠습니다. 

# In[25]:


# counts 컬럼에 동일하게 1 값을 넣어주기
tips["counts"] = 1
tips.head()


# 이제 `pivot_table`을 counts에 대한 sum(총합)을 이용하면 간단하게 각 시간/요일 별 데이터 수를 확인할 수 있겠죠!     

# In[27]:


# index 는 time 으로, columns 는 day 로 설정하여 피봇 테이블 만들기 
tips.pivot_table("counts", index = "time", columns = "day", aggfunc = "sum")


# 빈 값으로 나타난 것은 데이터가 없기 때문입니다. 0으로 채워주고, 데이터 또한 정수형으로 바꿔주도록 하겠습니다.      

# In[29]:


# 빈 값은 0으로 채워주고 정수형으로 형 변환해주기
# margins = True : All(합계) 추가
# fillna() : NaN 값을 인자로 채워준다. 빈도수 같은 경우 0으로 대체가 가능하다.
# astype : 형변환
tips.pivot_table("counts", index = "time", columns = "day", aggfunc = "sum", margins = True).fillna(0).astype("int64")


# In[30]:


# 비율 확인하기
# 금요일의 손님은 전체의 7.7% 밖에 되지 않는다.
(19 / 244) * 100


# 그래프로도 확인을 해보죠. 

# In[31]:


# 그래프로 시각화해보기
sns.countplot(data = tips, x = "day", hue = "time")
plt.show()


# ### (4) `size` : 테이블 인원수
# 
# size는 테이블 인원수를 나타냅니다.   
# 
# 테이블 인원 수에 따른 Tip 과 Total Bill 값을 시각화를 통해 확인해보도록 하겠습니다.

# In[38]:


# subplot 사용하여 1개 행, 2개 열로 두개의 컬럼 동시에 시각화하기 
plt.figure(figsize = (16,8))

plt.subplot(121)
sns.barplot(data = tips, x = "size", y = "tip")
plt.title("tip")

plt.subplot(122)
sns.barplot(data = tips, x = "size", y = "total_bill")
plt.title("total_bill")

plt.show()


# 잠깐! 여기서 그래프의 색상을 바꿀 수 있는 방법을 소개합니다.    
#     
# seaborn은 다양한 color palette를 제공하므로, 원하는 것을 골라서 사용할 수 있습니다. 
# - https://seaborn.pydata.org/generated/seaborn.color_palette.html    

# In[46]:


sns.set_palette("Set2")
# sns.set_palette("Paired")
# sns.set_palette("husl")


# pallete를 `husl`로 설정하면 그래프가 어떻게 바뀔까요?

# In[47]:


# 그래프 색상 바꿔서 다시 출력해보기 
plt.figure(figsize = (16,8))

plt.subplot(121)
sns.barplot(data = tips, x = "size", y = "tip")
plt.title("tip")

plt.subplot(122)
sns.barplot(data = tips, x = "size", y = "total_bill")
plt.title("total_bill")

plt.show()


# #### size별 데이터 개수는 어떨까?

# size별 데이터는 몇 개씩 있을까요? 확인해봅시다.

# In[48]:


# 테이블 인원수 데이터 개수 확인해보기
size_count = tips.groupby('size').size().reset_index(name = "counts")
size_count


# In[49]:


# 확인한 데이터 개수 시각화해보기
sns.countplot(data = tips, x = "size")
plt.show()


# 1인~6인 테이블이 존재하지만, 2인 테이블이 가장 많은 것을 확인할 수 있습니다.     

# ---

# # 3. 가설을 세우고 그에 따라 세부 분석해보기

# 지금까지 데이터를 어느정도 살펴보았으니, 이제 원하는 목표를 위해 가설을 세운 후 세부적으로 분석을 해보겠습니다.     

# ## 가설 : 테이블의 인원수를 나타내는 `size` 컬럼은 tip을 결정짓는 중요한 요소일 것이다

# 앞서 데이터를 살펴보면서 `size`가 `total_bill`과 `tips`를 결정짓는 중요한 요소임을 확인하였습니다.     
# 그렇다면 실제로 그런지 조금 더 구체적으로 살펴보죠!

# In[50]:


# size 에 따른 tip, tip_rate 그래프 시각화하기
plt.figure(figsize = (16,8))

plt.subplot(121)
sns.barplot(data = tips, x = "size", y = "tip")
plt.title("tip")

plt.subplot(122)
sns.barplot(data = tips, x = "size", y = "tip_rate")
plt.title("tip_rate")

plt.show()


# #### 그렇다면 `total_bill`과 `tip_rate`의 관계는 어떨까?

# `total_bill`과 `tip_rate`의 관계를 확인할 수 있는 그래프도 한 번 확인해보겠습니다.    

# In[58]:


# order=1 인 경우: 1차 함수(추세선)
# n차 함수의 n이 높아질수록 과적합(오버피팅) 문제가 발생한다. 주어진 데이터에만 작동하는 모델이 되어버린다.
sns.lmplot(data=tips, x="total_bill", y="tip_rate", fit_reg=True, order=1)
plt.show()


# In[59]:


# order=2 인 경우: 2차 함수
sns.lmplot(data=tips, x="total_bill", y="tip_rate", fit_reg=True, order=2)
plt.show()


# 어떤가요? 같은 그래프이지만 추세선은 다르게 나타납니다.    

# #### 이상치(Outlier)를 살펴보자

# 위 그래프를 통해 `total_bill`과 `tip_rate`는 반비례하는 경향을 가지는 것을 확인할 수 있습니다.    
# 
# 또 한 가지 흥미로운 점은, 혼자서 엄청 떨어져 있는 특이한 tip_rate(70%이상, 40%이상) 가 있다는 점입니다. 
# 이를 시각화를 통해 확인해보겠습니다.

# In[60]:


# 시각화로 이상치 확인해보기
# 이상치를 확인할 땐 seaborn의 boxplot을 많이 사용한다.
# boxplot은 통계지식이 필요하다. 박스 내의 선이 중앙값, 윗변이 상위 25%, 하랫번이 상위 75%의 값이다.
# 윗변은 q1 일사분위 수, 아랫변은 q3의 삼사분위 수 라고 부른다.
# 위의 꼬리는 q1에서 1.5x일사분위 수, 아랫 꼬리는 1.5x삼사분위 수 이다.
# 이상치 내의 값은 위 꼬리 ~ 아래 꼬리 까지이다.

plt.figure(figsize=(3,6))

sns.boxplot(data = tips, y = "tip_rate")
plt.show()


# 박스플롯을 그리면 데이터의 분포가 나타납니다.     
# 
# 이상치를 어떻게 처리하느냐는 상황에 따라, 도메인에 따라 달라질 수 있습니다. 박스플롯에 나타나는 모든 데이터포인트를 삭제할수도, 일부분을 삭제할수도 있습니다. 저희는 이상치 중에서도 혼자 가장 멀리 떨어져있는 70% 이상의 데이터포인트를 확인해보겠습니다.

# In[61]:


# 70% 이상의 팁 비율(이상치) 확인하기
# 두 명이 온 손님
tips[tips["tip_rate"] >= 70]


# 가장 멀리 떨어져있는 위 행만 삭제를 하고 진행해보도록 하겠습니다. 
# 
#  
# index를 확인했으니, pandas의 `drop` 함수로 간단히 제거할 수 있습니다

# In[62]:


# drop 함수로 이상치 제거하기
tips = tips.drop(172)
print(tips.shape)


# 이제  다시 분포를 확인해봅시다. `hue` 인자에 `time`을 넣어 확인해 보겠습니다.

# In[64]:


# 이상치 제거 이후 분포 다시 확인해보기 
sns.lmplot(data=tips, x="total_bill", y="tip_rate", fit_reg=True, order=1)
plt.show()


# 이제 전체적으로 혼자 엄청 떨어진 데이터 없이 정상적인 데이터 분포를 확인할 수 있고, 반비례 관계를 가지는 것을 확인했습니다.

# #### 그렇다면, `tip_rate`가 작은  큰 테이블이 좋은가, 아니면 `tip_rate`가 큰 작은 테이블이 좋은가?
# 이런 질문을 던져보지 않을 수 없겠죠! 바로 확인해봅시다.

# 먼저 우리가 원하는 컬럼인 `size`를 기준으로 `groupby`를 한 후, `describe` 함수를 이용해서 간단한 통계량을 한 번에 확인해보겠습니다.    
# `describe`는 전체 데이터에 대한 통계량을 한 번에 볼 때 유용합니다.     

# In[70]:


# describe 함수로 통계량 확인해보기
tips.groupby('size').describe()["total_bill"]


# 이런 결론을 내릴 수 있습니다.
# > **size가 큰 테이블은 작은 테이블보다 `tip_rate`가 작더라도 `tip` 금액은 클 것이다** 

# ### [가설] size는 tip을 결정짓는 가장 중요한 요소이다 : 결론
# 
#   
# `tip_rate`가 높더라도 `size`가 작다면 총 금액이 작아서, 비율로 환산할 시 실제로 받는 절대적인 tip 금액은 작다.   
# 그러므로, 일단은 **1순위로 size가 큰 테이블을 잡는 것이 유리하다.**

# ### 결론을 낸 후의 고찰

# 하지만, size를 다시 한 번 상기해보면 어떤가요?

# In[71]:


# size 다시 한번 시각화해보기
sns.countplot(data = tips, x = "size")
plt.show


# 실제로 size는 2명인 테이블이 154개로 가장 압도적이고, 1명이거나 5명, 6명인 테이블은 각각 전체의 5%도 안되는 개수를 가집니다.    
# 큰 테이블을 잡는 것이 유리하나, 그 빈도는 높지 않을 것임을 예상할 수 있습니다.

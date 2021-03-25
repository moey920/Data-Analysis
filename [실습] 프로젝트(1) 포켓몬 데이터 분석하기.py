
# coding: utf-8

# # [실습] 프로젝트 (1) : 포켓몬 데이터 분석하기 
# ## 포켓몬 데이터 뭉치에서 전설 포켓몬을 골라낼 수 있을까?

# 데이터 분석 프로젝트를 통한 탐색적 데이터 분석(EDA) 연습하기

# ### 학습 목표
# - 속성, 스탯 등 여러가지 데이터가 있는 포켓몬 데이터셋을 활용해서 `전설 포켓몬`의 특징을 파악합니다.
# - 데이터 분포를 확인하고, 변수간 관계를 파악하는 EDA를 통해 데이터 전체를 명확히 이해합니다.
# - EDA를 진행하는 과정에서 다양한 파이썬 함수 사용, 시각화 방법, 그래프 해석 방법을 학습합니다. 

# ---

# # Contents
# **1. [데이터 분석 준비하기](#1.-데이터-분석-준비하기)**   
# 
# **2. [EDA #1: 전설 포켓몬 데이터 셋 분리하기](#2.-EDA-#1:-전설-포켓몬-데이터-셋-분리하기)**   
# 
# 
# **3. [EDA #2: 모든 컬럼 뜯어보기](#3.-EDA-#2:-모든-컬럼-뜯어보기)**

# ---

# ## 1. 데이터 분석 준비하기

# 본격적으로 데이터 분석 프로젝트를 시작하기에 앞서, 먼저 앞으로의 데이터 분석에 필요한 module을 import 해 봅니다.    
# 1장에서 사용했던 데이터 분석을 위한 파이썬 라이브러리인 `numpy`, `pandas`, `matplotlib`, `seaborn` 을 사용해 보겠습니다.

# ### 1-1. Import Modules

# In[1]:


# numpy
import numpy as np

# pandas
import pandas as pd

# seaborn
import seaborn as sns

# matplotlib의 pyplot
import matplotlib.pyplot as plt

# matplotlib 시각화 결과를 jupyter notebook에서 바로 확인하기 위한 코드 작성 
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1-2. Load Dataset

# 모듈 임포트를 완료하였다면, 이제 이번 프로젝트에서 사용할 포켓몬 데이터셋을 가져와 보겠습니다. 
# 
# 데이터 출처 : [Kaggle(캐글)](https://www.kaggle.com/abcsds/pokemon) 
# 
# 엘리스에서는 이번 프로젝트에서 사용할 포켓몬 데이터 셋을 이미 실습 파일 목록에 함께 업로드하였기 때문에    
# 별개로 저장 및 업로드를 하시지 않아도 실습 진행이 가능합니다. 

# **참고 - Dataset Description 살펴보기**

# ```
# This data set includes 721 Pokemon, including their number, name, first and second type, and basic stats: HP, Attack, Defense, Special Attack, Special Defense, and Speed. It has been of great use when teaching statistics to kids. With certain types you can also give a geeky introduction to machine learning.
# 
# 이 데이터 셋에는 번호, 이름, 첫 번째 및 두 번째 유형 및 HP, 공격력, 방어력, 특수 공격, 특수 방어 및 속도와 같은 기본 통계를 포함한 721 개의 포켓몬이 포함되어 있습니다. 이 데이터 셋은 아이들에게 통계를 가르칠 때 매우 유용하며, 특정 유형의 경우 기계 학습(머신 러닝)에 대한 소개를 제공할 수 있습니다.
# 
# This are the raw attributes that are used for calculating how much damage an attack will do in the games. This dataset is about the pokemon games (NOT pokemon cards or Pokemon Go).
# 
# 게임 내에서 공격이 얼마나 많은 피해를 줄지 계산하는 데 사용되는 기본 속성입니다. 이 데이터 셋은 포켓몬 게임에 관한 것입니다 (포켓몬 카드 또는 포켓몬 고에 관한 데이터는 아닙니다.)
# 
# The data for this table has been acquired from several different sites, including:
# 
# pokemon.com
# pokemondb
# bulbapedia
# 
# 이 테이블의 데이터는 다음을 포함하여 여러 다른 사이트에서 얻었습니다.
# 
# One question has been answered with this database: The type of a pokemon cannot be inferred only by it's Attack and Deffence. It would be worthy to find which two variables can define the type of a pokemon, if any. Two variables can be plotted in a 2D space, and used as an example for machine learning. This could mean the creation of a visual example any geeky Machine Learning class would love.
# 
# 이 데이터베이스에 대한 질문과 그에 대한 답은 다음과 같습니다 : 포켓몬의 유형은 공격과 방어만으로 유추 할 수 없습니다. 포켓몬의 유형을 정의 할 수있는 두 변수를 찾는 것이 좋습니다. 2D 공간에 2 개의 변수를 시각화 할 수 있으며 이는 기계 학습(머신 러닝)의 예시로 사용됩니다. 이것은 머신 러닝 수업에서 좋아할만한 시각적 인 예를 만들어내는 것을 의미 할 수 있습니다.
# ```
# 

# 본격적으로 파이썬 라이브러리인 `pandas`를 활용해 데이터를 가져와봅니다.

# In[2]:


# read_csv로 csv 파일을 불러옵니다.pandas를 활용해 데이터를 가져와보도록 하겠습니다.
data = pd.read_csv('Pokemon.csv') 
pkemon = data # 오리지날 데이터셋은 따로 저장해놓는다.

# 데이터프레임의 크기를 확인합니다.
print(pkemon.shape)

# 데이터프레임 상위 5개 값 확인
pkemon.head()


# 전체 데이터는 **800개**로 800마리의 포켓몬 데이터가 있고, 각 포켓몬의 속성은 총 **13개**로 구성되어있음을 shape 함수를 통해 확인할 수 있습니다.

# #### 각 컬럼에 대한 설명
# 각 컬럼이 나타내는 바는 다음과 같습니다.  
# 
# 변수명 | 의미 
# - | - 
# **#** | <center>포켓몬 Id number</center> 
# **Name** | <center>포켓몬의 이름</center> 
# **Type 1** | <center>첫 번째 속성</center>   
# **Type 2** | <center>두 번째 속성</center> 
# **Total** | <center>전체 6가지 스탯의 총합</center> 
# **HP** | <center>포켓몬의 체력</center> 
# **Attack** | <center>물리공격력 (scratch, punch 등)</center>  
# **Defense** | <center>물리공격에 대한 방어력</center> 
# **Sp. Atk** | <center>특수공격력 (fire blast, bubble beam 등)</center>  
# **Sp. Def** | <center>특수공격에 대한 방어력.</center> 
# **Speed** | <center>포켓몬 매치에 대해 어떤 포켓몬이 먼저 공격할지를 결정.<br>(더 높은 포켓몬이 먼저 공격한다)</center>  
# **Generation** | <center>포켓몬의 세대. 현재 데이터에는 6세대까지 있다.</center>  
# **Legendary** | <center>전설의 포켓몬 여부. **!! Target feature !!**</center>  

# ### 1-3. 데이터 기본 확인 및 전처리하기

# In[3]:


# info 확인하기
pkemon.info()


# 결측치를 제거해야 할지 여부를 결정하기 위해 `Type 2` 컬럼을 조금 더 자세히 살펴보겠습니다. 

# #### 결측치 확인하기

# In[5]:


# 컬럼별 결측치 한번 더 확인해보기
# Type2만 true 값이 존재하는 것을 알 수 있다.
# pkemon.isnull()
pkemon.isnull().sum() # Type2 컬럼만 386개의 결측값이 나옴을 확인한다.


# In[9]:


# 직접 결측된 데이터를 눈으로 확인해보기
pkemon[pkemon['Type 2'].isnull()].head()


# - Type 2가 결측되어 있어도 다른 컬럼에 대한 정보는 모두 채워져 있기 때문에 바로 제거하자는 주장을 펼치기엔 위험 가능성이 너무 크겠군요!

# ---

# ## 2. EDA #1: 전설 포켓몬 데이터 셋 분리하기

# 전설 포켓몬과 일반 포켓몬 데이터를 분리하여 각각 다른 변수에 저장해보겠습니다.    
# 전설 포켓몬인지에 대한 정보가 저장되어 있는 컬럼, 즉 `Legendary`(전설의 포켓몬인지 아닌지의 여부) 값을 활용하면 
# 쉽게 분리할 수 있습니다.

# In[17]:


# Legendary 컬럼 값이 True 인 경우 legendary 변수에 저장하기
# 처리 후 reset_index()를 통해 기존의 인덱스를 제거하고 새로운 인덱스를 부여한다
legendary = pkemon[pkemon['Legendary'] == True].reset_index(drop=True)
# print(legendary)

# 데이터프레임의 크기를 확인합니다.
# .reset_index() 사용시 기존의 인덱스가 새로운 column으로 들어간 것을 볼 수 있다. 
# 그래서 (65,13)에서 (65,14)로 컬럼이 늘어났다.
# 따라서 .reset_index(drop=True) drop 인자를 통해 기존의 인덱스를 제거하는 과정을 거친다. (65, 13)
print(legendary.shape)

# 데이터프레임 상위 5개 값 확인
legendary.head()


# - 전체 데이터 개수가 800개였던 것을 기억해보면, 전설 포켓몬은 800개 중 65개밖에 존재하지 않음을 확인했습니다. 

# 전체 데이터 개수 중 전설 포켓몬 데이터의 비율을 간단한 코드 작성을 통해 확인해보겠습니다. 

# In[22]:


# 전체 데이터 중 전설 포켓몬 데이터의 비율을 출력해봅니다.
# 65, 800이 들어간다.
print("전체 데이터 중 전설 포켓몬 데이터 비율 : {}% ".format((legendary.shape[0]/pkemon.shape[0])*100))


# 나머지 일반 포켓몬 데이터를 분리해보겠습니다. 

# In[23]:


# 일반 포켓몬 데이터를 분리하여 변수에 저장하기
ordinary = pkemon[pkemon['Legendary'] == False].reset_index(drop=True)

# 데이터프레임 크기 확인하기
print(ordinary.shape)

# 상위 5개 값 확인하기
ordinary.head()


# ## 3. EDA #2: 모든 컬럼 뜯어보기

# 그럼 이제 본격적으로 데이터 내에 존재하는 각 컬럼을 하나씩 살펴보도록 하겠습니다.    
# 
# 먼저 전체 컬럼 이름을 확인해보겠습니다. 

# In[28]:


# pkemon 데이터셋 컬럼 출력하기
print(len(pkemon.columns))
pkemon.columns


# - 전체 13개 컬럼이 존재하고, 각 컬럼명을 list로 확인해보았습니다. 

# ### (1) 첫 번째 컬럼  `#` : id number

# In[31]:


# 총 몇개의 #(id number)값이 있는지 확인해보기, 고유값만 찾기
len(pkemon['#'].unique())


# - 전체 데이터는 총 800개인데 `#` 컬럼값 그보다 작은 **721개**의 데이터를 가집니다.     

# In[33]:


# 같은 #(id number)값을 가지는 데이터 빈도 수 확인을 통해 알아보기
# 데이터 빈도 수는 value_counts() 활용
pkemon['#'].value_counts()


# - #(id number) 479번 데이터가 6개 있네요, 이 데이터를 출력해보며 실제로 눈으로 확인해보겠습니다.   

# In[34]:


# id number 479번 확인하기
# 물, 불, 바람등의 속성만 다른 같은 포켓몬이 나온다.
print(pkemon[pkemon['#']==479])


# - `Rotom` 이라고 하는 동일한 이름에서 `RotomHeat` , `RotomWash`, `RotomFrost` 와 같은 단어들이 앞에 붙어 있는 포켓몬들이 동일한 id number로 존재하는 것을 확인할 수 있습니다. 
# - 다른 중복값도 한번 살펴볼까요?

# In[35]:


# id number 386번 확인하기
print(pkemon[pkemon['#']==386])


# - id number 가 386번인 경우도 동일한 단어들이 이름 내에 포함되어 있는 것을 확인할 수 있습니다.
# - 동일한 이름이면 같은 id number를 가진다. 하지만 모두 확인한 것은 아니기 때문에 확신할 수는 없다. 다만 전설 포켓몬 유무를 확인하기엔 부적합한 컬럼이라는 것을 알 수 있다.

# ### (2) 두 번째 컬럼 `Name` : 이름

# In[37]:


# 총 몇 개의 이름이 있는지 확인해보기
len(pkemon['Name'].unique())


# - 이름 컬럼의 고유 값의 개수는 데이터셋 전체 크기와 동일한 800개로, 이는 모든 포켓몬의 이름이 동일하지 않음을 의미합니다. 

# #### 특정 단어가 들어가있는 이름

# In[41]:


# 바로 확인해보자

# 이름이 비슷한 전설의 포켓몬들의 모임 names
# concat : 데이터 병합
n1, n2, n3, n4, n5 = legendary[3:6], legendary[14:24], legendary[25:29], legendary[46:50], legendary[52:57]
names = pd.concat([n1, n2, n3, n4, n5]).reset_index(drop=True)
print(names)


# 먼저, 위에서 중복된 #(id number)컬럼을 통해 확인했듯이 전설의 포켓몬 중에는 이름이 한 이름에서 파생되어 만들어진 이름들이 있죠. 

# In[42]:


# 전설 포켓몬 중 이름이 세트로 지어져있는 포켓몬들의 모임 set_names
sn1, sn2 = names[:13], names[23:]
set_names = pd.concat([sn1, sn2]).reset_index(drop=True)
set_names


# 어떤가요? 이들은 모두 세트로 이름이 지어져 있습니다.    
# - **"MewTwo", "Latias", "Latios", "Kyogre", "Groudon", "Rayquaza", "Kyurem"** 등의 이름에서부터 그 앞에 성이 붙여진다.
# - 따라서 포켓몬 원형이 전설 포켓몬일 경우 해당 포켓몬의 성이 붙으면 그 포켓몬도 전설 포켓몬이다 라는 것을 알 수 있습니다.

# #### 긴 이름

# 특정 단어가 포함되어있는지 여부 뿐만 아니라 이름의 길이는 어떨까요? 
# 데이터셋에 이름 길이 컬럼을 생성해서 비교해보도록 합니다.

# apply를 활용하여 이름의 길이를 반환하여 name_count라는 이름의 새로운 컬럼으로 추가해주도록 하겠습니다. 

# In[43]:


# legendary에 이름 길이 컬럼 생성
legendary['name_count'] = legendary['Name'].apply(lambda i : len(i))
legendary.head()


# In[44]:


# ordinary에 이름 길이 컬럼 생성
ordinary['name_count'] = ordinary['Name'].apply(lambda i : len(i))
ordinary.head()


# 그렇다면 새롭게 추가한 이 `name_count` 컬럼은 어떤 특징을 갖게 될까요? 직접 그래프로 그려 시각화해보도록 하겠습니다. 

# In[45]:


# 새롭게 추가한 이름 길이 컬럼 시각화
# 위, 아래 그래프를 나눠 표현하기 위해 subplot을 사용한다.

# 도화지 생성
plt.figure(figsize=(18,10))

# 211 : 행의 숫자, 열의 숫자, 그리고자 하는 그래프의 순서
plt.subplot(211)
sns.countplot(data = legendary, x = "name_count")
plt.title("Legendary")

plt.subplot(212)
sns.countplot(data = ordinary, x = "name_count")
plt.title("Ordinary")


# - 그래프를 통해 확인한 결과 **전설의 포켓몬은 16 이상의 긴 이름을 가진 포켓몬이 많은** 반면, **일반 포켓몬은 10 이상의 길이를 가지는 이름의 빈도가 아주 낮습니다.**
# - 다만 y값(빈도)가 다르기 때문에 확률로 보는 것이 더 적당하다

# In[46]:


# 전설의 포켓몬의 이름이 10 이상일 확률
print(round(len(legendary[legendary["name_count"] > 9]) / len(legendary) * 100, 2), "%")


# In[47]:


# 일반 포켓몬의 이름이 10 이상일 확률
print(round(len(ordinary[ordinary["name_count"] > 9]) / len(ordinary) * 100, 2), "%")


# - **전설의 포켓몬의 이름이 10 이상일 확률은 41%** 를 넘는 반면, **일반 포켓몬의 이름이 10 이상일 확률은 약 16%** 밖에 안됨을 확인할 수 있습니다!   
# - 이는 아주 큰 차이이므로 legendary인지 아닌지를 구분하는데에 큰 의미를 가집니다.

# ---

# ### (3) 세,네 번째 컬럼 `Type 1` & `Type 2` : 포켓몬의 속성

# 3번째, 4번째 컬럼인 Type 1과 Type 2 컬럼은 동일하게 포켓몬의 속성을 나타내고 있기 때문에 같이 살펴보겠습니다.    
# 두 마리의 포켓몬을 한 번 뽑아볼까요?

# In[48]:


# 두 개의 데이터를 확인해보겠습니다.
pkemon.loc[[6, 10]]


# - 포켓몬이 가지는 속성은 **기본적으로 하나, 또는 최대 두 개까지** 가질 수 있는 것 같군요.  

# 그렇다면, 속성의 종류는 총 몇 가지인지 알아봅시다.

# In[52]:


# 속성의 종류를 확인해봅시다.
print(len(pkemon['Type 1'].unique()), len(pkemon['Type 2'].unique()))


# In[54]:


# 2번 속성이 더 많은 이유는 무엇일까요? Type 2는 NaN(결측값)이 있기 때문이다.
set(pkemon['Type 2']) - set(pkemon['Type 1'])


# `Nan`값임을 알 수 있고, 따라서 그 외의 나머지 18가지 속성은 같은 종류로 데이터가 들어가 있음을 알 수 있습니다.

# In[60]:


# 모든 타입을 types 변수에 저장
types = list(pkemon['Type 1'].unique())
print(len(types))
print(types)


# 그렇다면 Type를 하나만 가지고 있는 포켓몬은 몇 마리일까요?

# In[61]:


# Type 2가 NaN값인 데이터의 개수
len(pkemon[pkemon['Type 2'].isna()])


# 총 386개의 포켓몬은 속성을 하나만 가지고, 나머지는 두 개의 속성을 가지는군요!
# 
# 그렇다면 Type을 두개 모두 가지고 있을 때 전설 포켓몬일 확률이 높은지도 함께 계산해볼까요?

# In[63]:


# 전설 포켓몬 중 두개의 Type을 모두 가지고 있는 경우
legendary['Type 2'].notnull().sum()


# #### `Type 1` 데이터 분포 plot
# 일반 포켓몬과 전설 포켓몬의 속성 분포가 각각 어떤지 확인해보겠습니다.      

# In[64]:


# Type 1 분포 시각화하기
plt.figure(figsize=(18,10))

plt.subplot(211)
# hue : Legendary 값에 따라 분리된 그래프를 얻을 수 있다.
sns.countplot(data = pkemon, x = "Type 1", hue = "Legendary", order = types)
plt.title("All Pokemon")

plt.subplot(212)
sns.countplot(data = legendary, x = "Type 1", order = types)
plt.title("Legendary Pokemon")


# 그렇다면, 피벗테이블로 각 속성에 전설 포켓몬들이 몇 퍼센트씩 있는지 확인해봅시다.

# In[65]:


pd.pivot_table(pkemon, index="Type 1", values="Legendary").sort_values(by=["Legendary"], ascending=False).T


# #### `Type 2` 데이터 분포 plot
# Type 2는 어떨까요?    
# 참고로, Type 2에는 NaN(결측값)이 존재했었습니다. Countplot을 그릴 때에는 결측값은 자동으로 제외됩니다.

# In[66]:


# Type 2 분포 시각화하기
plt.figure(figsize=(18,10))

plt.subplot(211)
# hue : Legendary 값에 따라 분리된 그래프를 얻을 수 있다.
sns.countplot(data = pkemon, x = "Type 2", hue = "Legendary", order = types)
plt.title("All Pokemon")

plt.subplot(212)
sns.countplot(data = legendary, x = "Type 2", order = types)
plt.title("Legendary Pokemon")


# Type 2 또한 일반 포켓몬과 전설 포켓몬의 분포 차이가 보입니다.    
# 
# 
# 역시 피벗 테이블로도 확인해볼까요?

# In[67]:


pd.pivot_table(pkemon, index="Type 2", values="Legendary").sort_values(by=["Legendary"], ascending=False).T


# ### (4) `Total` : 모든 스탯의 총합

# 이번 실습에서 사용하는 데이터셋에 존재하는 포켓몬은 포켓몬의 체력인 **HP**, 물리공격력 **Attack**, 물리공격에 대한 방어력인 **Defense** , 특수공격력 (fire blast, bubble beam 등)인 **Sp. Atk** ,  특수공격에 대한 방어력인 **Sp. Def** , 포켓몬 매치에 대해 어떤 포켓몬이 먼저 공격할지를 결정(더 높은 포켓몬이 먼저 공격한다)하는 **Speed** 의 총 6가지의 스탯값을 가집니다. 
# 
# 그리고 여기서 살펴볼 Total 컬럼은 이 6가지 속성값의 총 합입니다.

# In[68]:


# 6가지 전체 스탯을 stats 변수에 저장
stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]


# #### `Total`값에 따른 분포 plot
# 그렇다면 Total값과 전설 포켓몬과는 어떤 관계가 있는지, 해당 컬럼에 따른 전설 포켓몬의 특징을 확인해봅시다.

# In[70]:


# 분포를 산점도로 표현하여 시각화해보기
plt.figure(figsize=(16,8))

plt.scatter(data = pkemon, x = "Type 1", y = "Total")
plt.scatter(data = legendary, x = "Type 1", y = "Total")
plt.show()


# 먼저 전설 포켓몬들의 속성 Total값을 확인해봅시다.

# In[73]:


# 전설 포켓몬 속성 Total 값 확인하기
plt.figure(figsize=(8,6))

plt.scatter(data = legendary, y = "Type 1", x = "Total")
plt.show()
# 전설 포켓몬의 속성 총 합은 특정 수치에 몰려있는 것을 알 수 있다.


# 데이터 시각화를 통해 특징이 보이는 것 같습니다.
# 
# 실제로 전설의 포켓몬이 가지는 Total값들의 고유값을 확인해봅시다.

# In[75]:


# 전설 포켓몬이 가지는 Total 고유값 확인하기
print(legendary["Total"].unique())
len(legendary["Total"].unique())


# 실제로 단 9가지 값밖에 존재하지 않는군요! 그래프로도 확인해 봅시다.

# In[76]:


# 전설 포켓몬이 가지는 Total값에 대한 수 확인
plt.figure(figsize=(8,6))

sns.countplot(data = legendary, x = "Total")
plt.show()


# 총 65마리의 전설 포켓몬이 9개의 Total값만 가진다는 것은,

# In[77]:


# Total 스탯값이 같은 전설 포켓몬 개수 확인
65/9


# **약 7.22마리끼리는 같은 Total 스탯값을 가진다**는 의미와 같습니다. 이는 언뜻 봐도 꽤.. 높은 값인 것 같은 느낌을 주네요.

# 그렇다면 일반 포켓몬은 어떨까요? 같은 방법으로 다시 확인해봅시다.

# In[78]:


# 일반 포켓몬이 가지는 Total 고유값 확인하기
print(ordinary["Total"].unique())
len(ordinary["Total"].unique())


# In[81]:


# 일반 포켓몬이 가지는 Total값에 대한 개수 시각화를 통해 확인하기
plt.figure(figsize=(12,6))

sns.countplot(data = ordinary, x = "Total")
plt.show()


# 일반 포켓몬은 총 195개의 Total 속성값을 가지고, 전체 일반 포켓몬은 (800 - 65), 즉 735마리이므로,

# In[80]:


# Total 스탯값이 같은 일반 포켓몬 개수 확인
735/195


# **약 3.77마리만 같은 Total 스탯값을 가지는군요.**   

# ### (5) Stats: `HP`, `Attack`, `Defense`, `Sp. Atk`, `Sp. Def`, `Speed`

# 그렇다면 총합인 Total 뿐만 아니라 각각의 stat에 대해서는 어떻게 분포되어 있을까요?    
# subplot으로 여러 그래프를 한 번에 확인해봅시다.

# In[82]:


# 6가지 스탯 값 시각화하기 
plt.figure(figsize=(16,8))

plt.subplot(231)
plt.scatter(data=pkemon, y="Total", x="HP")
plt.scatter(data=legendary, y="Total", x="HP")
plt.title('HP')

plt.subplot(232)
plt.scatter(data=pkemon, y="Total", x="Attack")
plt.scatter(data=legendary, y="Total", x="Attack")
plt.title('Attack')

plt.subplot(233)
plt.scatter(data=pkemon, y="Total", x="Defense")
plt.scatter(data=legendary, y="Total", x="Defense")
plt.title('Defense')

plt.subplot(234)
plt.scatter(data=pkemon, y="Total", x="Sp. Atk")
plt.scatter(data=legendary, y="Total", x="Sp. Atk")
plt.title('Sp. Atk')

plt.subplot(235)
plt.scatter(data=pkemon, y="Total", x="Sp. Def")
plt.scatter(data=legendary, y="Total", x="Sp. Def")
plt.title('Sp. Def')

plt.subplot(236)
plt.scatter(data=pkemon, y="Total", x="Speed")
plt.scatter(data=legendary, y="Total", x="Speed")
plt.title('Speed')

plt.show()


# - **`HP`, `Defense`, `Sp. Def`**
# - **`Attack`, `Sp. Atk`, `Speed`**

# ### (6) `Generation` : 포켓몬의 세대

# Generation은 각 포켓몬의 "세대"를 나타냅니다.    
#    
# 각 세대에 대한 포켓몬의 수를 확인해 봅시다.

# In[83]:


# 세대별 데이터 시각화 
plt.figure(figsize=(8,6))

plt.subplot(211)
sns.countplot(data = pkemon, x = "Generation", hue = "Legendary")
plt.title('All pokemon')

plt.subplot(212)
sns.countplot(data = legendary, x = "Generation")
plt.title('Legendary pokemon')


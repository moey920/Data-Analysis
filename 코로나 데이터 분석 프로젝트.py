
# coding: utf-8

# ### [Kaggle 데이터셋]
# ---
# 
# 
# # Data Science for COVID-19 (DS4C)
# ### 캐글 코리아와 함께하는 2nd ML 대회 - House Price Prediction

# - 링크 : https://www.kaggle.com/kimjihoo/coronavirusdataset

# ### 지도 데이터 커널
# - https://www.kaggle.com/mbnb8317/ds4c-tutorial-all-about-folium

# ### Import Modules

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import warnings
warnings.filterwarnings("ignore")


# ### Load Dataset

# In[27]:


patient_info = pd.read_csv('coronavirusdataset/Patientinfo.csv')
print(patient_info.shape)
patient_info.head(3)


# ## 1. 다른 사람을 가장 많이 전염시킨 사람은 누굴까?

# #### (1) groupby에서 행의 개수를 세기 위해 `count` 컬럼 생성하기

# In[31]:


patient_info['count'] = 1
patient_info.head(2)


# #### (2) `infected_by` 로 groupby 시킨 후 `sum` 으로 행의 개수 확인하기

# In[37]:


patient_info.groupby('infected_by').sum().head(3)


# In[41]:


top_infection = patient_info.groupby('infected_by').sum().sort_values('count', ascending=False)
top_infection.head(5)


# # (3) index 새로 생성하기

# In[43]:


top_infection = top_infection.reset_index()
top_infection.head(5)


# #### (4) `top_infection`에서 필요없는 컬럼은 모두 제외하고, `infected_by`, `count` 컬럼만 가져오기
#  두 개 이상의 컬럼을 가져올 때는 Series가 아니기 때문에 df[[col1, col2]]를 해야한다.
#  df에서 컬럼들을 인덱싱하면 df가 된다

# In[46]:


top_infection.shape
top_infection = top_infection[['infected_by', 'count']]
top_infection.head(5)


# - 가장 많이 감염시킨 `2000000205` 환자의 경우 51명을 감염시켰습니다.

# ## 2. 다른 사람을 가장 많이 접촉한 사람들은 어떤 특징이 있을까?

# #### (1) 다른 사람을 가장 많이 접촉한 사람은 `contact_number`로 sorting 해서 확인할 수 있다.

# In[49]:


top_contacter = patient_info.sort_values('contact_number', ascending=False)
top_contacter.head(5)


# - 1등은 1160명, 2등은 1091명(10대!) 등으로, 이들에는 어떤 특징이 있을까? 
#     - 신천지? or 교인? 
#     - 중국인? 
#     - 대구? 부산? 광주?

# ## 3. 신천지의 치사율은 일반인의 치사율보다 낮을까?

# #### (1) 전체 치사율을 확인해보자

# - `deceased` (사망) 의 상태인 환자의 수

# In[56]:


patient_info.head(2)
patient_info['state'].unique()
# 마스킹 : Ture값만 반환
the_deceased = patient_info[patient_info['state'] == 'deceased']
the_deceased.shape
len(the_deceased)


# - 전체 환자의 수

# In[57]:


len(patient_info)


# - 전체 치사율

# In[68]:


round(78 / 5158 * 100, 2)


# - 약 1.51%

# #### (2) 신천지의 `infection_case` 확인

# In[62]:


patient_info['infection_case'].unique()


# - 위의 전체 케이스 중 신천지는 `Shincheonji Church`에 해당

# #### (3) 신천지 데이터프레임 생성

# In[65]:


shincheonji = patient_info[patient_info['infection_case'] == 'Shincheonji Church']
shincheonji.head(5)
shincheonji.shape


# - 신천지로 감염된 경우는 총 107명

# #### (4) 신천지에서 사망한 사람의 수

# In[66]:


shincheonji[shincheonji['state'] == 'deceased']


# #### (5) 신천지의 치사율

# In[67]:


round(2 / 107 * 100, 2)


# - 약 1.87%, 조금 낮긴 하지만 데이터셋이 매우 작기때문에 본 가설이 유의하다고 볼 수는 없다.

# ## 4. `Case.csv` 파일 다뤄보기
# - `Case.csv` 파일은 감염의 케이스에 대한 정보가 담겨있다.

# In[70]:


case = pd.read_csv('coronavirusdataset/Case.csv')
print(case.shape)
case.head(2)


# #### (1) case를 `confirmed`로 정렬해본다면?

# In[71]:


case.sort_values(by='confirmed', ascending=False).head(10)


# - 신천지가 가장 많고, 2위는 '다른 사람과 접촉', '기타' 로 집단감염에 해당하지 않는 기타 데이터이므로 주의해야 함
# - 4위 또한 신천지로, 1위는 대구 신천지, 4위는 경상북도 신천지임을 확인할 수 있다.

# #### (2) case에서 신천지만 뽑아본다면?

# In[74]:


case[case['infection_case'] == 'Shincheonji Church']


# - 각 지역별 신천지의 감염 사례가 전부 담겨있음

# #### (3) case를 `infection_case`로 묶어본다면?

# In[78]:


infection_case = case.groupby('infection_case').sum()
infection_case = infection_case.sort_values('confirmed', ascending=False)
infection_case.head(5)


# - 1위는 역시나 신천지
# - 2위, 3위, 4위는 모두 집단감염이 아닌 개인감염에 해당
# - 그 밑은 클럽, 병원, 콜센터 등 집단 감염

# ## 5. `TimeAge.csv` 데이터 다뤄보기, 연령대 별 감염 추이

# `TimeAge.csv`의 데이터로 `timeage`라는 데이터 프레임 생성한다.

# In[81]:


timeage = pd.read_csv('coronavirusdataset/TimeAge.csv')
print(timeage.shape)
timeage.head(10)


# - `timeage` 데이터프레임은 날짜별로 각 연령대의 누적 확진자를 담고 있음.  
# - 따라서 마지막 날짜의 데이터만 확인하면 됨.

# #### (1) 마지막 9개 행만 잘라서 가져오기

# In[87]:


last_timeage = timeage[-9:]
last_timeage


# #### (2) `deceased` / `confirmed`로 치사율 컬럼 만들기

# In[91]:


last_timeage['mortality'] = round((last_timeage['deceased'] / last_timeage['confirmed']) * 100, 2)
last_timeage


# #### (3) 간단하게 그래프 그려보기

# In[93]:


import seaborn as sns
sns.lineplot(data = last_timeage, x = 'age', y = 'mortality')


# - 80대로 올라갈수록 치사율이 급격하게 높아진다

# ## 6. 나이대 별 완치 비율은 어떨까?

# #### (1) `state`가 `released`인 데이터를 모아보자, 단 100세의 데이터는 한 명 밖에 없으므로 비교 대상에서 제외하자

# In[111]:


patient_info = patient_info[patient_info.age != '100s']
released = patient_info[patient_info["state"] == "released"]
print(released.shape)
released.head()


# #### (2) 위 데이터를 `age`로 묶어서 몇 명인지 세보자

# In[112]:


release_count = released.groupby('age').sum()
release_count['release_count'] = release_count["count"]
release_count


# #### (3) 나이대 별 완치자 수를 그래프로 그려보자
# - 그래프의 x축 순서를 `order` 인자로 직접 넣어주자

# In[113]:


sns.countplot(data=released, 
              x = 'age', 
              order = ['0s', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']) #order : x축의 순서 지정


# - 20대가 가장 많다!
# - 하지만 이는 절대적인 완치자 수이므로, 우리는 '확진자 수 대비 완치자 수'라는 수치인 '완치율'을 확인해 볼 필요가 있다.

# #### (4) 확진자 수 : 확진 된 사람들의 연령 별 분포도 확인하기 위해 전체 데이터인 `patient_info`를 `age`로 묶어서 `confirm_count`로 저장하자

# In[114]:


confirm_count = patient_info.groupby('age').sum()
confirm_count["confirm_count"] = confirm_count["count"]
confirm_count


# #### (4) 완치 된 사람들의 수

# In[117]:


release_count['release_count']


# #### (5) 확진 된 사람들의 수

# In[116]:


confirm_count["confirm_count"]


# #### (6) 위 두 가지 표를 붙여주고 싶다! `concat`으로 붙여주자. 다만, 가로로 붙일 거니까 `axis=1`을 꼭 넣자!

# In[118]:


age_count = pd.concat([release_count['release_count'], 
                       confirm_count["confirm_count"]], axis=1).fillna(0).astype('int64')
age_count


# #### (7) 완치율이라는 `release_rate`를 만들어주자

# In[119]:


age_count["release_rate"] = round(age_count["release_count"] / age_count["confirm_count"] * 100, 2)
age_count


# #### (8) 그래프를 그려보자

# In[120]:


sns.barplot(data=age_count, x=age_count.index, y='release_rate')
plt.show()


# - 완치율은 0s~50s 까지 크게 변화가 없다가, 60s부터 급격히 감소한다. 특히 80s 이상은 완치율이 20% 언저리로 매우 낮다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

print(train_data.info())

train_corr=train_data.corr()
print(train_corr["Survived"].sort_values(ascending=False)) # ascending이 False이므로 내림차순 정렬

def show_chart(feature):
    print("a")
    survived = train_data[train_data["Survived"]==1][feature].value_counts() # train_data에 Survived가 1인 사람들에서 feature의 종류에 따라
    # LIST의 형태로 반환 하는듯.
    print(survived)
    dead = train_data[train_data["Survived"]==0][feature].value_counts()
    data_frame = pd.DataFrame([survived,dead])
    data_frame.first_valid_index = ['Survived','Dead']
    data_frame.plot(kind='bar',stacked=True,figsize=(10,5))

show_chart('Parch')
train_data = train_data.drop(['Cabin'],axis=1)
train_data = train_data.drop(['Ticket'],axis=1)
test_data = test_data.drop(['Cabin'],axis=1)
test_data = test_data.drop(['Ticket'],axis=1)

print("b")
print(train_data['Embarked'].value_counts())
# S 644 C 168 Q 77이네

train_data = train_data.fillna({"Embarked" : "S"})
# train_data에 비어있는 항목들을 모두 S로 채워주자.

embarked_mapping = {'S' : 0 , 'C' : 1 , 'Q' : 2}
train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)

print(train_data['Embarked'].head()) # 앞의 5개만 확인하는 것.
# 알파벳을 숫자로 할당해주자

combine = [train_data , test_data]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
print(pd.crosstab(train_data['Title'],train_data['Sex']))

# 이름을 따오고
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady' , 'Capt' , 'Col' , 'Don' , 'Dr' , 'Major' , 'Rev' , 'Jonkheer' , 'Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess' , 'Lady' , 'Sir'] , 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle' , 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme' , 'Mrs')

print(train_data[['Title' , 'Survived']].groupby(['Title'],as_index=False).mean())

# 이름을 통합시킴 . 비슷한 이름의 연관성을 찾기 위해서

title_mapping = {'Mr' : 1 , 'Miss' : 2 , 'Mrs' : 3 , 'Master' : 4 , 'Royal' : 5 , 'Rare' : 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Title은 문자니까 숫자로 바꿔줌

train_data = train_data.drop(['Name' , 'PassengerId'] , axis=1)
test_data = test_data.drop(['Name'] , axis=1)
combine = [train_data , test_data]

# name과 PassengerId를 drop시킴

sex_mapping = {'male' : 0 , 'female' : 1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# sex도 mapping해줌

train_data['Age'] = train_data['Age'].fillna(-0.5)
test_data['Age'] = test_data['Age'].fillna(-0.5)
bins = [-1,0,5,12,18,24,35,60,np.inf]
labels = ['Unknown' , 'Baby' , 'Child' , 'Teenager' , 'Student' , 'YoungAdult','Adult' , 'Senior']
train_data['AgeGroup'] = pd.cut(train_data['Age'],bins,labels=labels)
test_data['AgeGroup'] = pd.cut(test_data['Age'],bins,labels=labels)
show_chart('AgeGroup')
print("AgeGroup index",train_data['AgeGroup'])

# age를 일정 범위로 통합시킴.(학습을 위해).

age_title_mapping = {1: 'YoungAdult' , 2: 'Student' , 3: 'Adult' , 4: 'Baby' , 5: 'Adult' , 6: 'Adult'}

for x in range(len(train_data['AgeGroup'])):
    if train_data['AgeGroup'][x] == 'Unknown':
        train_data['AgeGroup'][x] = age_title_mapping[train_data['Title'][x]]

for x in range(len(test_data['AgeGroup'])):
    if test_data['AgeGroup'][x] == 'Unknown':
        test_data['AgeGroup'][x] = age_title_mapping[test_data['Title'][x]]

# 이름 즉 title을 보고 연령을 추론해서 넣어주기

age_mapping = {'Baby' : 1 , 'Child' : 2 , 'Teenager' : 3 , 'Student' :4 , 'YoungAdult' : 5 , 'Adult' : 6 , "Senior" : 7}
train_data['AgeGroup'] = train_data['AgeGroup'].map(age_mapping)
test_data['AgeGroup'] = test_data['AgeGroup'].map(age_mapping)


train_data = train_data.drop(['Age'],axis=1)
test_data = test_data.drop(['Age'],axis=1)

train_data['Fare'].fillna(train_data['Fare'].median() , inplace=True)
test_data['Fare'].fillna(train_data['Fare'].median(),inplace=True)

train_data['FareBand'] = pd.qcut(train_data['Fare'],4).cat.codes
test_data['FareBand'] = pd.qcut(test_data['Fare'],4).cat.codes

train_data = train_data.drop(['Fare'],axis=1)
test_data = test_data.drop(['Fare'],axis=1)

print(train_data.info())
print(test_data.info())

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

Y_value = train_data['Survived']
X_value = train_data.drop('Survived',axis=1)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10 , shuffle=True , random_state=0)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_value,Y_value)
scoring = 'accuracy'
score = cross_val_score(clf,X_value,Y_value , cv=k_fold , n_jobs=1 , scoring = scoring)
print(score)

print(round(np.mean(score)*100,2))

prediction = clf.predict(test_data)
submission = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : prediction
})
submission.to_csv('submission.csv' , index=False)

submission = pd.read_csv("submission.csv")
print(submission.head())

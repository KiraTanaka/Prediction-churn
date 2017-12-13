import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('dataset_train.csv')
id= data['id']
data = data.drop(['id'], axis=1)

def selection_of_parameters(classifier,params):
	return GridSearchCV(estimator=classifier, param_grid=params, cv=5)

def classification(model):
    model_fit = model.fit(X_train, y_train)
    y_pred=model_fit.predict(X_test)
    print(classification_report(y_test, y_pred))
    #return y_pred	
	
headers = list(data)
for header in headers:
    data.loc[data[header].isnull(), header] = 0
    data[header] = data[header].astype(int)
	
features = list(set(data.columns) - set('reason'))

#Матрица корреляций

corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
seaborn.heatmap(corr)
seaborn.plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train = scale(X_train)
X_test = scale(X_test)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train,y_train)

#KNeighbors

print('KNeighbors')

knn = KNeighborsClassifier()
params_knn = {'n_neighbors': (5,10,15),
				'weights': ('uniform','distance'),
				'algorithm' : ('ball_tree', 'kd_tree', 'brute', 'auto')}
model_knn = selection_of_parameters(knn,params_knn)
classification(model_knn)

#RandomForest

print('RandomForest')

rfc = RandomForestClassifier()
params_rfc = {'n_estimators': (5,10,15,20,30,50),
				'max_depth': (None,10,20,30,40),
				'min_samples_split': (3,2,5,10)}
model_rfc = selection_of_parameters(rfc,params_rfc)
classification(model_rfc)

#DecisionTree

print('DecisionTreeClassifier')

dtclf = DecisionTreeClassifier()
params_dtclf = {'criterion':('gini','entropy'),
				 'splitter' : ('best', 'random'),
				 'max_depth': (10,15,20,30,40,50),
				 'min_samples_split' : (5,10,15,20)}
model_dtclf = selection_of_parameters(dtclf,params_dtclf)
classification(model_dtclf)

#GaussianNB

print('GaussianNB')

gnb = GaussianNB()
classification(gnb)

#GradientTreeBoosting

print('GradientTreeBoosting')

gbclf = GradientBoostingClassifier()
parama_gbclf = {'n_estimators':range(20,81,10),
				'max_depth':(1,2,4,6,8,10),
				'min_samples_split':(10,20,30,40,50,80,100)}
model_gbclf = selection_of_parameters(gbclf,parama_gbclf)
classification(model_gbclf)


#seaborn.jointplot(x='feat_id', y='reason', data=data, kind='reg');
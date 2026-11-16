from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
model = RandomForestClassifier(
    n_estimators=5000, 
    max_depth=50, 
    min_samples_split=2,
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1
)
data = pd.read_csv('data.csv')
X = data
X = X.drop(columns=['quality'])
data['quality'] = data['quality'].map({
    3:0,
    4:0,
    5:0,
    6:1,
    7:2,
    8:2,
    9:2
})
Y = data['quality']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
prec = precision_score(y_test,pred,average='macro')
rec = recall_score(y_test,pred,average='macro')
f1 = f1_score(y_test,pred,average='macro')
print("ACCURACY: ",acc)
print("PRECISION: ",prec)
print("RECALL: ",rec)
print("F1: ",f1)
joblib.dump(model,'predictor.pkl')
import pandas as pd 
import matplotlib.pyplot as plt

import numpy as np 

df=pd.read_csv("laptop_data.csv")

df.shape

df.isnull().sum()

df.info()

df.sample(5)

df["TypeName"].value_counts()

df.drop(columns=["Unnamed: 0"],inplace=True)

df.shape

df["Ram"]=df["Ram"].str.replace("GB","")
df["Weight"]=df["Weight"].str.replace("kg","")

df

df["OpSys"].value_counts()

df["Ram"]=df["Ram"].astype('int32')
df["Weight"]=df["Weight"].astype('float32')

df

df["TouchScreen"]=df["ScreenResolution"].apply(lambda x:1 if "Touchscreen" in x else 0)
df["Ips"]=df["ScreenResolution"].apply(lambda x:1 if "IPS" in x else 0)

df.shape

new=df["ScreenResolution"].str.split("x",n=1,expand=True)

df["X_res"]=new[0]
df["Y_res"]=new[1]

df.head()

df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

df.head()

df["X_res"]=df["X_res"].astype('int32')
df["Y_res"]=df["Y_res"].astype('int32')

df.info()

df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')

df.drop(columns=["Inches"],inplace=True)

df.head(5)

df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
            

df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)

df.drop(columns=['Cpu','Cpu Name'],inplace=True)

df.head()

df["OpSys"].value_counts()


df['Memory'] = df['Memory'].astype(str)
df['Memory'] = df['Memory'].str.replace(r'\.0', '', regex=True)
df['Memory'] = df['Memory'].str.replace('GB', '', regex=False)
df['Memory'] = df['Memory'].str.replace('TB', '000', regex=False)


new = df['Memory'].str.split(r'\+', n=1, expand=True)

df['first'] = new[0].str.strip()
df['second'] = new[1].fillna('0')


df['Layer1HDD'] = df['first'].str.contains('HDD', case=False, na=False).astype(int)
df['Layer1SSD'] = df['first'].str.contains('SSD', case=False, na=False).astype(int)
df['Layer1Hybrid'] = df['first'].str.contains('Hybrid', case=False, na=False).astype(int)
df['Layer1Flash_Storage'] = df['first'].str.contains('Flash Storage', case=False, na=False).astype(int)


df['Layer2HDD'] = df['second'].str.contains('HDD', case=False, na=False).astype(int)
df['Layer2SSD'] = df['second'].str.contains('SSD', case=False, na=False).astype(int)
df['Layer2Hybrid'] = df['second'].str.contains('Hybrid', case=False, na=False).astype(int)
df['Layer2Flash_Storage'] = df['second'].str.contains('Flash Storage', case=False, na=False).astype(int)


df['first'] = df['first'].str.extract(r'(\d+)')[0].fillna('0')
df['second'] = df['second'].str.extract(r'(\d+)')[0].fillna('0')


df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)


df['HDD'] = df['first'] * df['Layer1HDD'] + df['second'] * df['Layer2HDD']
df['SSD'] = df['first'] * df['Layer1SSD'] + df['second'] * df['Layer2SSD']
df['Hybrid'] = df['first'] * df['Layer1Hybrid'] + df['second'] * df['Layer2Hybrid']
df['Flash_Storage'] = (
    df['first'] * df['Layer1Flash_Storage'] +
    df['second'] * df['Layer2Flash_Storage']
)


df.drop(
    columns=[
        'first', 'second',
        'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage',
        'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage'
    ],
    inplace=True
)

df.head()

df.drop(columns=["Memory"],inplace=True)

df.head()

df.select_dtypes(include='number').corr()["Price"]

df.drop(columns=["Hybrid","Flash_Storage"],inplace=True)

df.head()

df["Gpu brand"]=df["Gpu"].apply(lambda x:x.split()[0])

df["Gpu brand"].value_counts()

df = df[df['Gpu brand'] != 'ARM']

df.shape

df.drop(columns=["Gpu"],inplace=True)

df.head()

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

df['os'] = df['OpSys'].apply(cat_os)

df.drop(columns=['OpSys'],inplace=True)

df.sample(10)

X=df.drop(columns=["Price"])
y=np.log(df["Price"])

X

y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')



from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

step1 = ColumnTransformer(
    transformers=[
        (
            'col_tnf',
            OneHotEncoder(
                sparse_output=False,
                drop='first',
                handle_unknown='ignore'   # <-- Add this
            ),
            [0, 1, 7, 10, 11]
        )
    ],
    remainder='passthrough'
)

step2 = XGBRegressor(
    n_estimators=45,
    max_depth=5,
    learning_rate=0.5,
    random_state=42
)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

import pickle 

pickle.dump(df,open("df.pkl","wb"))

pickle.dump(pipe,open("Model.pkl","wb"))
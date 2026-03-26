
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🚀 LuxeLoop Complete Analytics Dashboard")

df = pd.read_csv("cleaned_dataset.csv")

df_enc = df.copy()
for col in df_enc.select_dtypes(include='object').columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

section = st.sidebar.radio("Navigation",[
"Overview","Descriptive Analytics","Classification","Regression","Clustering","Association Rules","Recommendations"
])

# Overview
if section=="Overview":
    st.metric("Users",len(df))
    st.metric("Avg Spend",int(df.max_spend.mean()))
    st.metric("Conversion %",round(df.luxe_loop_interest.mean()*100,2))

# Descriptive
if section=="Descriptive Analytics":
    st.plotly_chart(px.histogram(df,x="max_spend"))
    st.plotly_chart(px.box(df,x="city_tier",y="max_spend"))
    st.plotly_chart(px.pie(df,names="income_group"))

X = df_enc.drop("luxe_loop_interest",axis=1)
y = df_enc["luxe_loop_interest"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

# Classification
if section=="Classification":
    pred = clf.predict(X_test)
    st.write("Accuracy",accuracy_score(y_test,pred))
    st.write("Precision",precision_score(y_test,pred))
    st.write("Recall",recall_score(y_test,pred))
    st.write("F1",f1_score(y_test,pred))
    prob = clf.predict_proba(X_test)[:,1]
    fpr,tpr,_ = roc_curve(y_test,prob)
    st.plotly_chart(px.area(x=fpr,y=tpr))

# Regression
reg = RandomForestRegressor()
reg.fit(X_train,df_enc.loc[X_train.index,"max_spend"])

if section=="Regression":
    preds = reg.predict(X_test)
    st.plotly_chart(px.scatter(x=y_test,y=preds))

# Clustering
if section=="Clustering":
    clusters = KMeans(n_clusters=4).fit_predict(StandardScaler().fit_transform(X))
    df["cluster"]=clusters
    st.plotly_chart(px.scatter(df,x="max_spend",y="trust_score",color="cluster"))

# Association
if section=="Association Rules":
    dummies = pd.get_dummies(df[["income_group","city_tier","preowned_willingness"]])
    freq = apriori(dummies,min_support=0.1,use_colnames=True)
    rules = association_rules(freq,metric="confidence",min_threshold=0.5)
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])

# Recommendations
if section=="Recommendations":
    st.success("Target high income & high trust users")
    st.info("Use discounts for price sensitive segments")
    st.warning("Build authentication trust")

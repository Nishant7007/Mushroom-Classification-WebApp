import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def main():
	st.title("Mushroom Classification Web App")
	st.sidebar.title("Choose Options")
	st.markdown("Are your mushrooms edible? 🍄")

	@st.cache(persist=True)
	def load_data():
		data = pd.read_csv('Data/mushrooms.csv')
		label = LabelEncoder()
		for col in data.columns:
			data[col] = label.fit_transform(data[col])
		return data

	@st.cache(persist=True)
	def split(df):
		Y = df['class']
		X = df.drop(columns=['class'])
		X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=8)
		return X_train, X_test, Y_train, Y_test

	def plot_metrics(metrics_list):
		if('Confusion_Matrix' in metrics_list):
			st.subheader('Confusion Matrix')
			plot_confusion_matrix(model,X_test,Y_test,display_labels=class_names)
			st.pyplot()

		if('ROC Curve' in metrics_list):
			st.subheader('ROC Curve')
			plot_roc_curve(model,X_test,Y_test)
			st.pyplot()

		if('Precision-Recall Curve' in metrics_list):
			st.subheader('Precision-Recall Curve')
			plot_precision_recall_curve(model,X_test,Y_test)
			st.pyplot()

	df = load_data()
	X_train, X_test, Y_train, Y_test = split(df)
	class_names=['Edible','Poisonous']
	st.sidebar.subheader("Choose Classifier")
	Classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine", "Logistic Regression", "Random Forest"))

	if(Classifier == "Support Vector Machine"):
		st.sidebar.subheader("Choose Hyperparameters")
		C = st.sidebar.number_input("C (regularization Parameter)",0.01,10.0,step=0.01,key='c')
		kernel = st.sidebar.radio("Kernel",("rbf","Linear"),key='kernel')
		gamma = st.sidebar.radio("Gamma (Kernel Coefficient)",("scale", "auto"),key="gamma")

		metrics = st.sidebar.multiselect("Select metrics to plot",('Confusion_Matrix','ROC Curve','Precision-Recall Curve'))

		if(st.sidebar.button("Classify",key='Classify')):
			st.subheader("Support Vector Machine (SVM) Results")
			model = SVC(C=C,kernel=kernel,gamma=gamma)
			model.fit(X_train,Y_train)
			accuracy = model.score(X_test,Y_test)
			Y_pred = model.predict(X_test)
			st.write("Accuracy:",accuracy.round(2))
			st.write("Precision", precision_score(Y_test,Y_pred,labels=class_names).round(2))
			st.write("Recall", recall_score(Y_test,Y_pred,labels=class_names).round(2))
			plot_metrics(metrics)

	if(st.sidebar.checkbox("Show Raw Data",False)):
		st.subheader("Mushroom Dataset Classification")
		st.write(df)




if(__name__ == '__main__'):
	main()



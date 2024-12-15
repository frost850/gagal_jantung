
# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Analisis Data Kesehatan Jantung")

# Upload file CSV
file = st.file_uploader("Unggah file CSV", type=["csv"])

if file is not None:
    # Membaca dataset
    heart_data = pd.read_csv(file, delimiter=',')
    
    # Menampilkan informasi dataset
    st.write("Total data dalam dataset:", len(heart_data))
    st.write("\nInformasi dataset:")
    st.write(heart_data.info())
    
    # Menampilkan lima data pertama
    st.write("\nLima data pertama:")
    st.write(heart_data.head())
    
    # Memeriksa missing value
    st.write("\nMemeriksa missing value:")
    st.write(heart_data.isnull().sum())
    
    st.write("\nPersentase missing value:")
    st.write((heart_data.isnull().sum() / len(heart_data)) * 100, "%")
    
    # Memeriksa kolom dalam DataFrame
    if 'Risk_Outcome' in heart_data.columns:
        features = heart_data.drop(columns=['Risk_Outcome'])
        target = heart_data['Risk_Outcome']
        
        # Normalisasi fitur
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(features)
        normalized_data = pd.DataFrame(normalized_features, columns=features.columns)
        normalized_data['Risk_Outcome'] = target
        
        st.write(normalized_data.head())
        
        # Membagi data menjadi set pelatihan dan pengujian
        train_data, test_data = train_test_split(normalized_data, test_size=0.2, random_state=42)
        
        # Model KNN
        k = st.slider("Pilih nilai k untuk KNN", 1, 20, 3)
        knn_model = KNeighborsClassifier(n_neighbors=k)
        X_train = train_data.drop(columns=['Risk_Outcome'])
        y_train = train_data['Risk_Outcome']
        X_test = test_data.drop(columns=['Risk_Outcome'])
        y_test = test_data['Risk_Outcome']
        
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        
        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy KNN:", accuracy)
        st.write("Classification Report:\n", classification_report(y_test, y_pred))
        
        # Visualisasi Confusion Matrix
        st.write("Confusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.title("Confusion Matrix KNN")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
        
        # Model Logistic Regression
        log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
        log_reg_model.fit(X_train, y_train)
        y_pred_log = log_reg_model.predict(X_test)
        
        accuracy_log = accuracy_score(y_test, y_pred_log)
        st.write("Logistic Regression Accuracy:", accuracy_log)
        st.write("Classification Report:\n", classification_report(y_test, y_pred_log))
        
        # Visualisasi Confusion Matrix untuk Logistic Regression
        st.write("Confusion Matrix Logistic Regression:")
        conf_matrix_log = confusion_matrix(y_test, y_pred_log)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix_log, annot=True, fmt="d", cmap="Greens", ax=ax)
        plt.title("Confusion Matrix Logistic Regression")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
        
        # Model Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write("Random Forest Accuracy:", accuracy_rf)
        st.write("Classification Report:\n", classification_report(y_test, y_pred_rf))
        
        # Visualisasi Confusion Matrix untuk Random Forest
        st.write("Confusion Matrix Random Forest:")
        conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Oranges", ax=ax)
        plt.title("Confusion Matrix Random Forest")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
    else:
        st.error("Error: Kolom 'Risk_Outcome' tidak ditemukan dalam dataset.")
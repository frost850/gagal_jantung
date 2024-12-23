import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from io import StringIO

# Fungsi untuk menyimpan model
def save_model(model, filename):
    """Menyimpan model machine learning"""
    joblib.dump(model, filename)

# Fungsi untuk memuat model
def load_model(filename):
    """Memuat model machine learning yang telah disimpan"""
    return joblib.load(filename)

def main():
    # Custom CSS dengan desain yang lebih modern
    st.markdown("""
        <style>
        /* Custom CSS for Header */
        .main-header {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        /* Styling untuk judul */
        .title-text {
            color: #1E88E5;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        /* Card style untuk sections */
        .stCard {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            padding: 0.75rem 1.5rem;
            font-size: 1.1rem;
            font-weight: 500;
            border-radius: 5px;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            background-color: #1976D2;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Input fields styling */
        .stNumberInput>div>div>input {
            border-radius: 5px;
            border: 1px solid #E0E0E0;
            padding: 0.5rem;
        }
        
        /* Selectbox styling */
        .stSelectbox>div>div {
            border-radius: 5px;
            border: 1px solid #E0E0E0;
        }
        
        /* Prediction results styling */
        .prediction-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 5px solid #1E88E5;
            margin: 1rem 0;
        }
        
        .risk-high {
            color: #d32f2f;
            font-weight: bold;
        }
        
        .risk-low {
            color: #388e3c;
            font-weight: bold;
        }
        
        /* About page styling */
        .about-section {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header aplikasi
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<h1 class="title-text">Analisis Data Kesehatan Jantung</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    menu = ["🏠 Home", "🔍 Predict", "ℹ️ About"]
    choice = st.sidebar.radio("Navigation", menu)

    if choice == "🏠 Home":
        show_home_page()
    elif choice == "🔍 Predict":
        show_prediction_page()
    else:
        show_about_page()

def show_home_page():
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📊 Visualisasi Dataset dan Evaluasi Model")
    file = st.file_uploader("Unggah file CSV", type=["csv"])

    if file is not None:
        try:
            heart_data = pd.read_csv(file, delimiter=',', encoding='utf-8')
            heart_data = preprocess_data(heart_data)
            show_data_analysis(heart_data)
            train_and_evaluate_models(heart_data)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

def preprocess_data(data):
    """Melakukan preprocessing pada dataset"""
    data = data.fillna(data.mean())
    le = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])
    return data

def show_data_analysis(data):
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.write("### 📈 Data Overview")
    st.write(data.head())
    st.write("### ℹ️ Data Info")
    buffer = StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write("### 📊 Statistical Analysis")
    st.write(data.describe())
    st.markdown('</div>', unsafe_allow_html=True)
    show_enhanced_visualizations(data)

def show_enhanced_visualizations(data):
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.write("### 📊 Data Visualizations")
    
    # Correlation Matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    st.pyplot(plt)
    
    # Distribution plots
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

def train_and_evaluate_models(data):
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    X = data.drop('Risk_Outcome', axis=1)
    y = data['Risk_Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    save_model(scaler, 'scaler.joblib')
    
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }
    
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        save_model(model, f'{name.lower().replace(" ", "_")}_model.joblib')
        
        results[name] = {
            'cv_score': cv_scores.mean(),
            'test_score': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        show_model_results(name, results[name])
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_results(model_name, results):
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.write(f"### 🔍 {model_name} Results")
    st.write(f"Cross-validation Score: {results['cv_score']:.4f}")
    st.write(f"Test Score: {results['test_score']:.4f}")
    st.write("Classification Report:")
    st.text(results['classification_report'])
    st.markdown('</div>', unsafe_allow_html=True)

def show_prediction_page():
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("🔍 Prediksi Risiko Penyakit Jantung")
    
    try:
        scaler = load_model('scaler.joblib')
        models = {
            'KNN': load_model('knn_model.joblib'),
            'Logistic Regression': load_model('logistic_regression_model.joblib'),
            'Random Forest': load_model('random_forest_model.joblib')
        }
        
        input_data = get_user_input()
        
        if st.button("Prediksi"):
            input_scaled = scaler.transform(input_data)
            predictions = {}
            for name, model in models.items():
                pred = model.predict(input_scaled)
                pred_prob = model.predict_proba(input_scaled)
                predictions[name] = {'prediction': pred[0], 'probability': pred_prob[0]}
            show_predictions(predictions)
            
    except FileNotFoundError:
        st.error("⚠️ Model belum dilatih. Silakan latih model terlebih dahulu di halaman Home.")
    st.markdown('</div>', unsafe_allow_html=True)

def get_user_input():
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("🔢 Usia", min_value=0, max_value=100, value=50)
        creatinine_phosphokinase = st.number_input("🧪 Creatinine Phosphokinase", min_value=0, max_value=10000, value=200)
        ejection_fraction = st.number_input("💗 Ejection Fraction (%)", min_value=0, max_value=100, value=30)
        platelets = st.number_input("🔬 Platelets", min_value=0, max_value=500000, value=250000)
        serum_creatinine = st.number_input("🧪 Serum Creatinine", min_value=0.0, max_value=15.0, value=1.0)
    
    with col2:
        serum_sodium = st.number_input("🧪 Serum Sodium", min_value=100, max_value=150, value=130)
        time = st.number_input("⏰ Time (days)", min_value=0, max_value=365, value=10)
        anaemia = st.selectbox("🩸 Anemia", ['Tidak', 'Ya'])
        diabetes = st.selectbox("🍬 Diabetes", ['Tidak', 'Ya'])
        high_blood_pressure = st.selectbox("🩺 Tekanan Darah Tinggi", ['Tidak', 'Ya'])
        sex = st.selectbox("👤 Jenis Kelamin", ['Wanita', 'Pria'])
        smoking = st.selectbox("🚬 Merokok", ['Tidak', 'Ya'])
    
    input_data = pd.DataFrame({
        'age': [age],
        'anaemia': [1 if anaemia == 'Ya' else 0],
        'creatinine_phosphokinase': [creatinine_phosphokinase],
        'diabetes': [1 if diabetes == 'Ya' else 0],
        'ejection_fraction': [ejection_fraction],
        'high_blood_pressure': [1 if high_blood_pressure == 'Ya' else 0],
        'platelets': [platelets],
        'serum_creatinine': [serum_creatinine],
        'serum_sodium': [serum_sodium],
        'sex': [1 if sex == 'Pria' else 0],
        'smoking': [1 if smoking == 'Ya' else 0],
        'time': [time]
    })
    
    st.markdown('</div>', unsafe_allow_html=True)
    return input_data

def show_predictions(predictions):
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.markdown("### 📊 Hasil Prediksi")
    
    for model_name, pred in predictions.items():
        risk_class = "risk-high" if pred['prediction'] == 1 else "risk-low"
        confidence = max(pred['probability'])*100
        
        st.markdown(f"""
        <div class="prediction-card">
            <h4>{model_name}</h4>
            <p>Prediksi: <span class="{risk_class}">
                {'⚠️ Risiko Tinggi' if pred['prediction'] == 1 else '✅ Risiko Rendah'}</span></p>
            <p>Tingkat Keyakinan: <strong>{confidence:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_about_page():
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.markdown("""
    # 🏥 Tentang Aplikasi Analisis Kesehatan Jantung

    ### Deskripsi Aplikasi
    Aplikasi ini adalah sistem prediksi risiko penyakit jantung berbasis machine learning yang dirancang untuk membantu 
    mengidentifikasi potensi risiko penyakit jantung berdasarkan berbagai parameter kesehatan pasien.

    ### 📊 Fitur Utama:
    1. **Homepage (Halaman Utama)**
       - Memungkinkan pengguna mengunggah dataset kesehatan dalam format CSV
       - Menampilkan visualisasi data terperinci termasuk korelasi antar variabel
       - Menyajikan analisis statistik komprehensif dari dataset
       - Menampilkan performa model machine learning yang digunakan

    2. **Prediction (Halaman Prediksi)**
       - Input parameter kesehatan pasien seperti:
         * Usia, jenis kelamin, dan riwayat kesehatan
         * Parameter darah (Creatinine, Sodium, dll)
         * Faktor risiko seperti diabetes dan tekanan darah tinggi
       - Menghasilkan prediksi menggunakan 3 model berbeda:
         * K-Nearest Neighbors (KNN)
         * Logistic Regression
         * Random Forest
       - Menampilkan hasil prediksi dengan tingkat keyakinan

    ### 🔬 Teknologi yang Digunakan:
    - **Machine Learning**: Sklearn, Pandas, NumPy
    - **Visualisasi**: Matplotlib, Seaborn
    - **Interface**: Streamlit
    
    ### ⚕️ Catatan Penting:
    Aplikasi ini dikembangkan untuk tujuan edukasi dan penelitian. Hasil prediksi tidak boleh digunakan 
    sebagai pengganti diagnosis medis profesional.

    ### 👨‍💻 Pengembang:
    **Muhamad Farhan Ismail Dewanata**
    
    ### 📝 Versi: 1.0
    """)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

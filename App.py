import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Analisis Naive Bayes", layout="wide")

# Judul Aplikasi
st.title("🛍️ Analisis Intensi Pembeli Online (Naive Bayes)")
st.markdown("---")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    # Pastikan file csv berada di folder yang sama dengan app.py
    dataset = pd.read_csv('online_shoppers_intention.csv')
    return dataset

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'online_shoppers_intention.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()

# --- Preprocessing Awal (Sama seperti code sebelumnya) ---
# Kita copy dataframe agar data asli tidak berubah untuk display
df_processed = df.copy()
le = LabelEncoder()
df_processed['Month'] = le.fit_transform(df_processed['Month'])
df_processed['VisitorType'] = le.fit_transform(df_processed['VisitorType'])
df_processed['Weekend'] = le.fit_transform(df_processed['Weekend'])
df_processed['Revenue'] = le.fit_transform(df_processed['Revenue'])

# --- BAGIAN 1-4: Penjelasan Dataset ---
st.header("1. Eksplorasi Dataset")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tentang Dataset")
    st.write("""
    Dataset ini berisi data transaksi sesi pengguna di sebuah e-commerce. 
    Tujuannya adalah memprediksi **Intensi Pembelian (Revenue)** berdasarkan perilaku browsing pengguna.
    """)
    
    st.subheader("Informasi Data")
    # Poin 2 & 3: Bentuk dan Jumlah
    st.write(f"**Bentuk Data:** {df.shape[0]} Baris, {df.shape[1]} Kolom")
    st.write(f"**Total Data:** {df.shape[0]}")

with col2:
    st.subheader("Fitur (Kolom)")
    # Poin 4: Daftar Kolom
    st.write(df.columns.tolist())

st.write("**Sampel Data Asli:**")
st.dataframe(df.head())

# --- BAGIAN 5: Seleksi Fitur ---
st.markdown("---")
st.header("2. Seleksi Fitur & Preprocessing")

st.info("Sesuai rancangan kode, kita memilih 2 fitur utama untuk visualisasi 2D: **ProductRelated_Duration** dan **PageValues**.")

# Memilih fitur (Index 5 dan 8 sesuai request sebelumnya)
X = df_processed.iloc[:, [5, 8]].values
y = df_processed.iloc[:, -1].values # Kolom Revenue

# Menampilkan hasil seleksi
st.write("**Data setelah seleksi fitur (X):**")
df_selection = pd.DataFrame(X, columns=['ProductRelated_Duration', 'PageValues'])
st.dataframe(df_selection.head())

# Splitting & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# --- BAGIAN HASIL: Naive Bayes ---
st.markdown("---")
st.header("3. Hasil Analisis Naive Bayes")

# Training Model
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Prediksi
y_pred = classifier.predict(X_test)

# Metrik Evaluasi
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Tampilan Metrik
col_metric1, col_metric2 = st.columns(2)
with col_metric1:
    st.metric("Akurasi Model", f"{acc*100:.2f}%")

with col_metric2:
    st.write("**Confusion Matrix:**")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig_cm)

# --- Visualisasi Hasil (Training & Test) ---
st.subheader("Visualisasi Decision Boundary")

def visualize_streamlit(X_set, y_set, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    
    ax.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    
    ax.set_xlim(X1.min(), X1.max())
    ax.set_ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j, edgecolors='black')
    
    ax.set_title(title)
    ax.set_xlabel('ProductRelated_Duration (Scaled)')
    ax.set_ylabel('PageValues (Scaled)')
    ax.legend()
    return fig

# Pilihan Tab untuk Training vs Test
tab1, tab2 = st.tabs(["Hasil Training Set", "Hasil Test Set"])

with tab1:
    st.write("Visualisasi pola data pada **Training Set**:")
    with st.spinner('Membuat grafik Training Set... (mungkin butuh beberapa detik)'):
        fig_train = visualize_streamlit(X_train, y_train, 'Naive Bayes (Training set)')
        st.pyplot(fig_train)

with tab2:
    st.write("Visualisasi prediksi pada **Test Set**:")
    with st.spinner('Membuat grafik Test Set...'):
        fig_test = visualize_streamlit(X_test, y_test, 'Naive Bayes (Test set)')
        st.pyplot(fig_test)

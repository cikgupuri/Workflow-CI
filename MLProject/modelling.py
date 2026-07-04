import os
import glob
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Mengaktifkan Autolog MLflow
mlflow.sklearn.autolog()
mlflow.set_experiment("Eksperimen_Prediksi_Dropout_Maspuri")

def main():
    # Menentukan folder data secara absolut
    base_dir = r"C:\BELAJAR\PIJAK-DICODING\Proyek_Akhir_MSML\Eksperimen_SML_Maspuri-Andewi\data"
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    
    if not csv_files:
        print(f"Eror: Tidak ada file CSV ditemukan di folder: {base_dir}")
        return
    
    data_path = csv_files[0]
    print(f"Berhasil menemukan dataset: {os.path.basename(data_path)}")
    print("Memuat dataset dengan pemisah titik koma (;)...")
    
    # Menambahkan sep=';' agar kolom terbaca dengan benar
    df = pd.read_csv(data_path, sep=';')
    
    # Membersihkan tanda kutip satu jika ada di nama kolom
    df.columns = [col.replace("'", "").replace('"', '').strip() for col in df.columns]
    
    # Deteksi Otomatis Kolom Target
    target_column = None
    for col in ['Target', 'target', 'Status', 'status', 'Label', 'label']:
        if col in df.columns:
            target_column = col
            break
            
    if target_column is None:
        target_column = df.columns[-1]
        
    print(f"Menggunakan kolom target: '{target_column}'")
    
    # Pemisahan Fitur (X) dan Target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Membagi data menjadi 80% Latih dan 20% Uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Memulai Sesi Run MLflow
    with mlflow.start_run(run_name="Random_Forest_Script_Baseline"):
        print("Melatih model Random Forest Classifier...")
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        akurasi = accuracy_score(y_test, y_pred)
        
        print("\n--- EKSPERIMEN SCRIPT MODELLING SELESAI ---")
        print(f"Akurasi Model pada Data Uji: {akurasi * 100:.2f}%\n")
        print("Laporan Klasifikasi Detail:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
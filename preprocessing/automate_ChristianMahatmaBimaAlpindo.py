import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(input_path, output_root):
    # 1. Validasi apakah file mentah ada
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        return

    # 2. Buat path folder tujuan: preprocessing/datamotor_preprocessed
    output_dir = os.path.join(output_root, 'datamotor_preprocessed')
    
    # Perbaikan Utama: Membuat folder jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Folder {output_dir} berhasil dibuat.")

    # 3. Load Data
    df = pd.read_csv(input_path)

    # 4. Cleaning & Preprocessing (Logika sama seperti sebelumnya)
    df = df.drop_duplicates(keep='first')
    
    if 'ex_showroom_price' in df.columns:
        df['ex_showroom_price'] = df['ex_showroom_price'].fillna(df['ex_showroom_price'].median())

    if 'name' in df.columns:
        df['brand'] = df['name'].apply(lambda x: x.split(' ')[0])
        df = df.drop(columns=['name'])

    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    fitur_numerik = ['year', 'km_driven', 'ex_showroom_price', 'brand']
    available_features = [f for f in fitur_numerik if f in df.columns]
    df[available_features] = scaler.fit_transform(df[available_features])
    
    # 5. Split Data
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # 6. Simpan hasil ke folder datamotor_preprocessed
    df.to_csv(os.path.join(output_dir, 'motor_preprocessed.csv'), index=False)
    train_df.to_csv(os.path.join(output_dir, 'motor_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'motor_test.csv'), index=False)
    
    print(f"Berhasil! File disimpan di: {output_dir}")
    print(f"Total Data: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    # Sesuaikan dengan struktur folder di repository GitHub kamu
    input_file = "dataset_raw/BIKE DETAILS.csv"
    # Root folder untuk hasil preprocessing
    output_root = "preprocessing" 

    preprocess_data(input_file, output_root)


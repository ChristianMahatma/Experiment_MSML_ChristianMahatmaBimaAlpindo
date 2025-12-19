import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(input_path, output_dir):
    # 1. Cek apakah file ada
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        return

    # 2. Load Data
    df = pd.read_csv(input_path)

    # 3. Cleaning: Hapus Duplikat
    df = df.drop_duplicates(keep='first')

    # 4. Handling Missing Values: ex_showroom_price
    if 'ex_showroom_price' in df.columns:
        df['ex_showroom_price'] = df['ex_showroom_price'].fillna(df['ex_showroom_price'].median())

    # 5. Feature Engineering: Ambil Brand dari nama motor
    if 'name' in df.columns:
        df['brand'] = df['name'].apply(lambda x: x.split(' ')[0])
        df = df.drop(columns=['name'])

    # 6. Encoding: Mengubah teks ke angka
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # 7. Scaling: Menyamakan skala fitur numerik (Kecuali target selling_price)
    scaler = StandardScaler()
    # Fitur yang akan di-scale
    fitur_numerik = ['year', 'km_driven', 'ex_showroom_price', 'brand']
    available_features = [f for f in fitur_numerik if f in df.columns]
    
    df[available_features] = scaler.fit_transform(df[available_features])

    # 8. Splitting Data (Train 80%, Test 20%)
    # Target adalah selling_price
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gabungkan kembali untuk disimpan sebagai file utuh
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # 9. Simpan hasil ke folder output
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(os.path.join(output_dir, 'motor_preprocessed.csv'), index=False)
    train_df.to_csv(os.path.join(output_dir, 'motor_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'motor_test.csv'), index=False)
    
    print(f"Berhasil! File disimpan di folder: {output_dir}")
    print(f"Total Data: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    # Konfigurasi Path
    input_file = "namadataset_raw/BIKE DETAILS.csv"
    output_folder = "preprocessing/namadataset_preprocessing"
    
    preprocess_data(input_file, output_folder)
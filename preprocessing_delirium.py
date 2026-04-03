"""
=======================================================================
DATA MINING - TUGAS PREPROCESSING DATA DELIRIUM
Politeknik Statistika STIS
=======================================================================
Tujuan  : Melakukan preprocessing (data cleaning, data reduction,
          data transformation) pada dataset Delirium.
Dataset : data_delirium_latihan_1.csv (457 observasi, 29 variabel)
=======================================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# LANGKAH 0 — LOAD DATA
# =====================================================================
print("=" * 65)
print("PREPROCESSING DATA DELIRIUM")
print("=" * 65)

df_raw = pd.read_csv('data_delirium_latihan_1.csv')
print(f"\n[LOAD] Ukuran data mentah : {df_raw.shape[0]} baris × {df_raw.shape[1]} kolom")

# Buat salinan kerja
df = df_raw.copy()

# =====================================================================
# LANGKAH 1 — IDENTIFIKASI MASALAH DATA
# =====================================================================
print("\n" + "=" * 65)
print("LANGKAH 1 : IDENTIFIKASI MASALAH DATA")
print("=" * 65)

issues = []

# 1a. Nilai blank/spasi (diperlakukan sebagai missing)
cols_str = ['pendidikan','lamaperawatan','timedelirium',
            'PENSTATUSFUNGSION','GGNKOGNITIF','KATALBUMIN',
            'HIPOKSIA','GANGGUANMETABOLIK']

print("\n[1a] Nilai kosong/spasi per kolom:")
for c in cols_str:
    blanks = (df[c].astype(str).str.strip() == '').sum()
    if blanks > 0:
        pct = blanks / len(df) * 100
        print(f"     {c:<25}: {blanks:>4} ({pct:.1f}%)")
        issues.append((c, 'missing_blank', blanks))

# 1b. Nilai tanda '-' pada tgldelir (bukan tanggal, bukan blank)
dash_tgldelir = (df['tgldelir'].astype(str).str.strip() == '-').sum()
print(f"\n[1b] tgldelir bernilai '-' (noise) : {dash_tgldelir}")
dash_tglmask  = (df['tglmask'].astype(str).str.strip() == '-').sum()
print(f"     tglmask  bernilai '-' (noise)  : {dash_tglmask}")

# 1c. statusnikah nilai 2.7 (tidak konsisten — seharusnya integer)
val_anom = df[df['statusnikah'] == 2.7]['statusnikah'].count()
print(f"\n[1c] statusnikah = 2.7 (tidak konsisten): {val_anom}")
issues.append(('statusnikah', 'inconsistent_value', val_anom))

# 1d. lamaperawatan = '75' (potensi outlier)
df['lamaperawatan_num'] = pd.to_numeric(df['lamaperawatan'].astype(str).str.strip(), errors='coerce')
print(f"\n[1d] Outlier lamaperawatan (>60 hari): "
      f"{(df['lamaperawatan_num'] > 60).sum()} nilai")
print(f"     Outlier lamaperawatan (='0')    : "
      f"{(df['lamaperawatan_num'] == 0).sum()} nilai")

# 1e. Format tanggal tidak konsisten
print("\n[1e] Contoh format tanggal tidak konsisten (tglmask):")
sample_dates = df['tglmask'].unique()[:6]
for d in sample_dates:
    print(f"     '{d}'")

# 1f. Variabel kategorik dikode sebagai angka (perlu diperiksa range)
cat_cols = {'jeniskelamin':[1,2], 'statusnikah':[1,2,3],
            'pendidikan':['1','2','3','4','5'],
            'pekerjaan':[1,2,3,4], 'ruangrwt':[1,2,3,4],
            'masukdari':[1,2], 'pembiayaan':[1,2,3],
            'delirium':[0,1], 'kondisiplg':[1,2,3],
            'STATUSDELIRIUM':[0,1]}
print("\n[1f] Periksa nilai di luar range yang diharapkan (kategorik):")
for col, valid in cat_cols.items():
    if df[col].dtype in [float, np.float64]:
        uniq = df[col].dropna().unique()
    else:
        uniq = df[col].unique()
    outliers = [v for v in uniq if v not in valid and str(v).strip() != '']
    if outliers:
        print(f"     {col}: nilai tidak valid = {outliers}")
    else:
        print(f"     {col}: OK")

# 1g. Duplikasi baris
dup = df.duplicated().sum()
print(f"\n[1g] Duplikasi baris: {dup}")

# =====================================================================
# LANGKAH 2 — DATA CLEANING
# =====================================================================
print("\n" + "=" * 65)
print("LANGKAH 2 : DATA CLEANING")
print("=" * 65)

# 2a. Ganti blank/spasi menjadi NaN
for c in cols_str:
    df[c] = df[c].astype(str).str.strip()
    df[c] = df[c].replace('', np.nan)

print("\n[2a] Nilai blank/spasi → NaN")

# 2b. Ganti '-' pada tgldelir dan tglmask menjadi NaN
df['tgldelir'] = df['tgldelir'].astype(str).str.strip().replace('-', np.nan)
df['tglmask']  = df['tglmask'].astype(str).str.strip().replace('-', np.nan)
print("[2b] tanda '-' pada tgldelir/tglmask → NaN")

# 2c. Perbaiki statusnikah = 2.7 → mode (=1)
mode_stat = df['statusnikah'][df['statusnikah'].isin([1,2,3])].mode()[0]
df['statusnikah'] = df['statusnikah'].replace(2.7, mode_stat)
df['statusnikah'] = df['statusnikah'].astype(int)
print(f"[2c] statusnikah = 2.7 → diganti dengan modus ({int(mode_stat)})")

# 2d. Konversi kolom numerik bertipe string
# lamaperawatan
df['lamaperawatan'] = pd.to_numeric(
    df['lamaperawatan'].astype(str).str.strip(), errors='coerce')

# timedelirium
df['timedelirium'] = pd.to_numeric(
    df['timedelirium'].astype(str).str.strip(), errors='coerce')

# pendidikan
df['pendidikan'] = pd.to_numeric(
    df['pendidikan'].astype(str).str.strip(), errors='coerce')

# PENSTATUSFUNGSION, GGNKOGNITIF, KATALBUMIN, HIPOKSIA, GANGGUANMETABOLIK
for c in ['PENSTATUSFUNGSION','GGNKOGNITIF','KATALBUMIN','HIPOKSIA','GANGGUANMETABOLIK']:
    df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors='coerce')

print("[2d] Konversi tipe data string numerik → numeric")

# Drop kolom helper yang dibuat di langkah 1
df.drop(columns=['lamaperawatan_num'], inplace=True, errors='ignore')

# 2e. Tampilkan jumlah missing setelah cleaning
print("\n[2e] Jumlah missing value setelah cleaning:")
miss = df.isnull().sum()
miss = miss[miss > 0]
for col, n in miss.items():
    pct = n / len(df) * 100
    print(f"     {col:<25}: {n:>4} ({pct:.1f}%)")

# 2f. Imputasi missing value
# – Variabel numerik: imputasi dengan median
# – Variabel kategorik: imputasi dengan modus
numeric_cols   = ['lamaperawatan','timedelirium','pendidikan',
                  'PENSTATUSFUNGSION','GGNKOGNITIF','KATALBUMIN',
                  'HIPOKSIA','GANGGUANMETABOLIK']

print("\n[2f] Imputasi missing value:")
for c in numeric_cols:
    n_miss = df[c].isnull().sum()
    if n_miss > 0:
        if c in ['pendidikan','PENSTATUSFUNGSION','GGNKOGNITIF',
                 'KATALBUMIN','HIPOKSIA','GANGGUANMETABOLIK']:
            # Kategorik biner/ordinal → modus
            fill_val = df[c].mode()[0]
            strategy = "modus"
        else:
            fill_val = df[c].median()
            strategy = "median"
        df[c] = df[c].fillna(fill_val)
        df[c] = df[c].astype(int) if c != 'lamaperawatan' else df[c]
        print(f"     {c:<25}: {n_miss:>4} nilai → diisi {strategy} = {fill_val}")

# 2g. Drop kolom tanggal (tidak digunakan dalam analisis, hanya identifikasi)
date_cols = ['tglmask','tgldelir','tglplg']
df.drop(columns=date_cols, inplace=True)
print(f"\n[2g] Drop kolom tanggal (tidak relevan untuk analisis): {date_cols}")

# =====================================================================
# LANGKAH 3 — DETEKSI & PENANGANAN OUTLIER
# =====================================================================
print("\n" + "=" * 65)
print("LANGKAH 3 : DETEKSI & PENANGANAN OUTLIER (IQR Method)")
print("=" * 65)

cont_cols = ['usia','lamaperawatan','TIME','JUMLAHOBATBARU']

outlier_summary = {}
for c in cont_cols:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[c] < lower) | (df[c] > upper)).sum()
    outlier_summary[c] = {'Q1':Q1,'Q3':Q3,'IQR':IQR,'lower':lower,'upper':upper,'n_outlier':n_out}
    print(f"\n  Kolom  : {c}")
    print(f"  Q1={Q1}, Q3={Q3}, IQR={IQR}")
    print(f"  Batas  : [{lower:.2f}, {upper:.2f}]")
    print(f"  Outlier: {n_out} baris")

# Winsorizing (clip) outlier agar tidak kehilangan data
for c in cont_cols:
    info = outlier_summary[c]
    n_before = ((df[c] < info['lower']) | (df[c] > info['upper'])).sum()
    df[c] = df[c].clip(lower=info['lower'], upper=info['upper'])
    print(f"\n[Winsorize] {c}: {n_before} outlier di-clip ke [{info['lower']:.2f}, {info['upper']:.2f}]")

# =====================================================================
# LANGKAH 4 — DATA REDUCTION (Feature Selection)
# =====================================================================
print("\n" + "=" * 65)
print("LANGKAH 4 : DATA REDUCTION — Pemilihan Atribut")
print("=" * 65)

# Kolom yang dihapus karena:
# - 'lamaperawatan' redundan dengan 'TIME' (TIME = min(lamaperawatan, 14))
# - JENKELBARU redundan dengan jeniskelamin
# - KATUMURBARU redundan dengan usia/KATUMURBARU
redundant = ['JENKELBARU','KATUMURBARU']
df.drop(columns=redundant, inplace=True, errors='ignore')
print(f"\n[4a] Drop kolom redundan: {redundant}")
print("     - JENKELBARU  : duplikat informasi dari jeniskelamin")
print("     - KATUMURBARU : duplikat informasi dari usia (kategorisasi usia)")

print(f"\n[4b] Ukuran data setelah reduksi: {df.shape[0]} baris × {df.shape[1]} kolom")

# =====================================================================
# LANGKAH 5 — DATA TRANSFORMATION (Normalisasi)
# =====================================================================
print("\n" + "=" * 65)
print("LANGKAH 5 : DATA TRANSFORMATION — Normalisasi")
print("=" * 65)

# Min-Max normalization untuk variabel kontinu
minmax_cols = ['usia','lamaperawatan','TIME']

df_normalized = df.copy()
print("\n[5a] Min-Max Normalization → skala [0, 1]")
for c in minmax_cols:
    mn = df[c].min()
    mx = df[c].max()
    df_normalized[c + '_norm'] = ((df[c] - mn) / (mx - mn)).round(4)
    print(f"     {c:<20}: min={mn}, max={mx}")

# Z-score normalization untuk usia (contoh)
print("\n[5b] Z-score Normalization untuk 'usia'")
mu  = df['usia'].mean()
sig = df['usia'].std()
df_normalized['usia_zscore'] = ((df['usia'] - mu) / sig).round(4)
print(f"     usia → mean={mu:.2f}, std={sig:.2f}")

# =====================================================================
# LANGKAH 6 — DISCRETIZATION (Concept Hierarchy)
# =====================================================================
print("\n" + "=" * 65)
print("LANGKAH 6 : DISCRETIZATION — Pembentukan Kategori Usia")
print("=" * 65)

bins_usia = [0, 59, 69, 79, 200]
labels_usia = ['<60 (Pra-Lansia)', '60-69 (Lansia Muda)',
               '70-79 (Lansia Menengah)', '>=80 (Lansia Tua)']
df_normalized['usia_kategori'] = pd.cut(df_normalized['usia'],
                                         bins=bins_usia,
                                         labels=labels_usia,
                                         right=True)
print("\n[6a] Discretization usia:")
print(df_normalized['usia_kategori'].value_counts().sort_index().to_string())

# Discretization untuk TIME (lama perawatan yang dianalisis)
bins_time = [-1, 3, 7, 14]
labels_time = ['1-3 hari','4-7 hari','8-14 hari']
df_normalized['TIME_kategori'] = pd.cut(df_normalized['TIME'],
                                         bins=bins_time,
                                         labels=labels_time)
print("\n[6b] Discretization TIME (hari observasi):")
print(df_normalized['TIME_kategori'].value_counts().sort_index().to_string())

# =====================================================================
# LANGKAH 7 — RANGKUMAN & SIMPAN HASIL
# =====================================================================
print("\n" + "=" * 65)
print("LANGKAH 7 : RANGKUMAN PREPROCESSING")
print("=" * 65)

print(f"""
  Data mentah  : 457 baris × 29 kolom
  Data bersih  : {df.shape[0]} baris × {df.shape[1]} kolom
  (+ transformasi) : {df_normalized.shape[1]} kolom total

  Tindakan yang dilakukan:
  ─────────────────────────────────────────────────────
  [Data Cleaning]
  • Nilai blank/spasi pada 8 kolom → NaN
  • Tanda '-' pada tgldelir/tglmask → NaN
  • statusnikah = 2.7 (tidak valid) → diisi modus (1)
  • Konversi tipe data (string→numeric) untuk 8 kolom
  • Imputasi missing: median (lamaperawatan, timedelirium)
                       modus (pendidikan, binary vars)
  • Drop kolom tanggal (tglmask, tgldelir, tglplg)

  [Outlier Detection & Handling]
  • Metode IQR pada: usia, lamaperawatan, TIME, JUMLAHOBATBARU
  • Penanganan: Winsorizing (clip ke batas IQR)

  [Data Reduction]
  • Drop JENKELBARU (redundan dengan jeniskelamin)
  • Drop KATUMURBARU (redundan dengan usia)

  [Data Transformation]
  • Min-Max normalization: usia, lamaperawatan, TIME
  • Z-score normalization: usia
  • Discretization: usia (4 kategori), TIME (3 kategori)
  ─────────────────────────────────────────────────────
""")

print("\n[7] Info data hasil preprocessing:")
print(df.dtypes)
print("\nSample 5 baris pertama (data bersih):")
print(df.head().to_string())

# Simpan hasil
df.to_csv('data_delirium_clean.csv', index=False)
df_normalized.to_csv('data_delirium_transformed.csv', index=False)
print("\n✔ Data bersih disimpan   : data_delirium_clean.csv")
print("✔ Data transformasi disimpan: data_delirium_transformed.csv")

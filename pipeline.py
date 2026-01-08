import json
import pandas as pd
import re
import os

# =============================
# Config
# =============================
json_folder = r"F:\AA FILE MERCU AZKA\Capstone project\Capstone_Project\Output Dataset bu afi"
output_csv = "instagram_accounts_Bu_Afi.csv"

# =============================
# Utility Functions
# =============================
def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rec_texts", [])

def is_timestamp(text):
    # Mendeteksi format jam seperti 6.16, 22.52, 09:00
    return bool(re.match(r"^\d{1,2}[:.]\d{2}$", text.strip()))

def is_stat_label(text):
    return text.lower().strip() in ["posts", "followers", "following", "postingan", "pengikut", "mengikuti"]

# =============================
# Core Extraction Logic
# =============================
def extract_fields_v2(texts):
    # Bersihkan list
    texts = [t.strip() for t in texts if t and isinstance(t, str)]
    
    fields = {
        "Username": "", "Bio": "", "Posts": "", "Followers": "", "Following": ""
    }

    # 1. CARI ANCHOR (Statistik)
    stat_indices = [i for i, t in enumerate(texts) if is_stat_label(t)]
    
    if not stat_indices:
        # Fallback: Cari username saja
        for t in texts:
            if not is_timestamp(t) and re.match(r"^[A-Za-z0-9._]{3,}$", t):
                fields["Username"] = t
                break
        return fields

    first_stat_idx = stat_indices[0]
    last_stat_idx = stat_indices[-1]

    # 2. EKSTRAKSI USERNAME (Area Sebelum Statistik)
    candidate_area = texts[:first_stat_idx]
    for t in candidate_area:
        if is_timestamp(t): continue
        if re.match(r"^\d+$", t): continue # Skip angka murni
        if len(t) < 3: continue
        if re.match(r"^[A-Za-z0-9._]+$", t):
            fields["Username"] = t
            break 
            
    # 3. EKSTRAKSI ANGKA (Area Sekitar Anchor)
    stats_area_start = max(0, first_stat_idx - 5)
    stats_area_end = min(len(texts), last_stat_idx + 2)
    stats_area = texts[stats_area_start:stats_area_end]
    nums = [t.replace(",", "").replace(".", "") for t in stats_area if re.match(r"^[0-9,.]+[KMBkmb]?$", t) and not is_timestamp(t)]
    
    if len(nums) >= 3:
        fields["Posts"], fields["Followers"], fields["Following"] = nums[:3]

    # =========================================
    # 4. EKSTRAKSI BIO (UPDATED UNTUK INDO)
    # =========================================
    raw_bio_lines = texts[last_stat_idx + 1:]
    clean_bio = []
    
    for line in raw_bio_lines:
        line_lower = line.lower()
        
        # --- STOP WORDS (Tanda Bio Berakhir) ---
        
        # 1. Cek Tombol UI (Inggris & Indo)
        # Menambahkan: "ikuti balik", "kirim pesan", "kontak"
        if any(keyword in line_lower for keyword in [
            "follow", "message", "contact", "email", "loading", "story", # ENG
            "ikuti", "kirim pesan", "kontak", "bagikan" # IDN
        ]):
            break
        
        # 2. Cek "Followed by" / "Diikuti oleh"
        # Regex ini menangkap:
        # "followed by", "followedby", "diikuti oleh", "diikutioleh"
        if re.search(r"(followed|diikuti)\s*(by|oleh)", line_lower):
            break
            
        # 3. Cek "and X others" / "dan X lainnya"
        # Menangkap pola: "dan 12 lainnya", "and 5 others"
        if re.search(r"(and|dan)\s+\d+\s+(others|lainnya)", line_lower):
            break

        # --- SKIP NOISE ---
        if re.match(r"^\d+$", line): continue # Skip angka sampah

        clean_bio.append(line)

    fields["Bio"] = " ".join(clean_bio).strip()

    return fields
# =============================
# Process ALL JSON files
# =============================
def process_all_json(json_folder, out_csv):
    rows = []
    
    if not os.path.exists(json_folder):
        print(f"Error: Folder {json_folder} not found.")
        return

    for file in os.listdir(json_folder):
        if file.lower().endswith(".json"):
            full_path = os.path.join(json_folder, file)
            # print("Processing JSON:", file) # Uncomment for debug

            try:
                texts = load_texts(full_path)
                fields = extract_fields_v2(texts) # Menggunakan Logic V2
                fields["SourceFile"] = file
                rows.append(fields)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    df = pd.DataFrame(rows)
    
    # Reorder columns for neatness
    cols = ["Username", "Bio", "Posts", "Followers", "Following", "SourceFile"]
    df = df[cols]
    
    df.to_csv(out_csv, index=False)
    print("\n" + "="*30)
    print(f"DONE! CSV Saved to: {out_csv}")
    print("="*30)
    print(df.head(10))
    return df

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    process_all_json(json_folder, output_csv)
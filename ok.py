import streamlit as st
import pandas as pd
import joblib
import re
import io
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords as nltk_stopwords
import joblib
import requests
from io import BytesIO
import json
import nltk
from functools import lru_cache
nltk.download('stopwords')

st.set_page_config(page_title="Klasifikasi Kendala SIPEDULI", layout="wide")

@st.cache_resource
def load_model_from_hf(url):
    response = requests.get(url)
    response.raise_for_status()  # Pastikan request berhasil
    model = joblib.load(BytesIO(response.content))
    return model

hf_model_url = "https://huggingface.co/mesakhbesta/clfsipeduli/resolve/main/voting_clf.joblib"

@st.cache_resource
def load_tfidf(path):
    return joblib.load(path)

@st.cache_resource
def load_svd(path):
    return joblib.load(path)

model = load_model_from_hf(hf_model_url)
tfidf = load_tfidf("tfidf.joblib")
svd = load_svd("svd.joblib")

@st.cache_data(show_spinner=False)
def get_stemmed_mapping(text_list):
    # load cache statis sekali saja
    with open("stem_cache.json", 'r') as f:
        static_cache = json.load(f)
    
    stemmer = StemmerFactory().create_stemmer()
    stop_factory = StopWordRemoverFactory()
    combined_stopwords = set(stop_factory.get_stop_words() + nltk_stopwords.words('english'))
    
    runtime_cache = {}  # untuk simpan kata baru selama runtime fungsi ini
    
    def clean_and_stem(t):
        t = str(t).lower()
        t = re.sub(r'[^a-z\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()

        # subs (optional)
        subs = {
            r"\bmasuk\b": "login",
            r"\blog-in\b": "login",
            r"\brenbis\b": "rencana bisnis",
            r"\baro\b": "administrator responsible officer",
            r"\bro\b": "responsible officer",
            r"\bmengunduh\b": "download",
            r"\bunduh\b": "download"
        }
        for pat, rep in subs.items():
            t = re.sub(pat, rep, t)

        words = [w for w in t.split() if w not in combined_stopwords]

        stemmed_words = []
        for w in words:
            if w in static_cache:
                stemmed_words.append(static_cache[w])
            elif w in runtime_cache:
                stemmed_words.append(runtime_cache[w])
            else:
                stemmed = stemmer.stem(w)
                runtime_cache[w] = stemmed
                stemmed_words.append(stemmed)

        return ' '.join(stemmed_words)

    mapping = {}
    for text in text_list:
        mapping[text] = clean_and_stem(text)

    return mapping

    
def preprocess_text(text):
    def remove_dear_ojk(t):
        return re.sub(r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK", "", t, flags=re.IGNORECASE) if isinstance(t, str) else t

    def extract_complaint(t):
        if not isinstance(t, str):
            return "Bagian komplain tidak ditemukan."
        # pola awal dan akhir
        start_p = [r"PERHATIAN: E-mail ini berasal dari pihak di luar OJK.*?attachment.*?link.*?yang terdapat pada e-mail ini."]
        end_p = [r"(From\s*.*?From|Best regards|Salam|Atas perhatiannya|Regards|Best Regards|Mohon\s*untuk\s*melengkapi\s*data\s*.*tabel\s*dibawah).*,?",
                 r"From:\s*Direktorat\s*Pelaporan\s*Data.*"]
        m_start = None
        for pat in start_p:
            matches = list(re.finditer(pat, t, re.DOTALL|re.IGNORECASE))
            if matches:
                m_start = matches[-1].end()
        if m_start:
            t = t[m_start:].strip()
        for pat in end_p:
            m = re.search(pat, t, re.DOTALL|re.IGNORECASE)
            if m:
                t = t[:m.start()].strip()
        # hapus sensitive info
        sens = [
            r"Nama\s*Terdaftar\s*.*",
            r"Email\s*.*",
            r"No\.\s*Telp\s*.*",
            r"User\s*Id\s*/\s*User\s*Name\s*.*",
            r"No\.\s*KTP\s*.*",
            r"Nama\s*Perusahaan\s*.*",
            r"Nama\s*Pelapor\s*.*",
            r"No\.\s*Telp\s*Pelapor\s*.*",
            r"Internal",
            r"Dengan\s*hormat.*",
            r"Jenis\s*Usaha\s*.*",
            r"Keterangan\s*.*",
            r"No\.\s*SK\s*.*",
            r"Alamat\s*website/URL\s*.*",
            r"Selamat\s*(Pagi|Siang|Sore).*",
            r"Kepada\s*Yth\.\s*Bapak/Ibu.*",
            r"On\s*full_days\s*\d+\d+,\s*\d{4}-\d{2}-\d{2}\s*at\s*\d{2}:\d{2}.*",
            r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK.*",
            r"No\.\s*NPWP\s*Perusahaan\s*.*",
            r"Aplikasi\s*OJK\s*yang\s*di\s*akses\s*.*",
            r"Yth\s*.*",
            r"demikian\s*.*",
            r"Demikian\s*.*",
            r"Demikianlah\s*.*"
            ]
        for pat in sens:
            t = re.sub(pat, "", t, flags=re.IGNORECASE)
        return t or "Bagian komplain tidak ditemukan."

    # Bersihkan kalimat umum
    def clean_email_text(t):
        pats = [
            r"(?i)(terlampir|mohon\s*bantuan|terima\s*kasi|yth|daripada\s*\w+\s*\@.*?\.com)",
            r"(?i)Selamat\s*.*",
            r"(?i)Dear\s*(Bapak/Ibu\s*)?Helpdesk\s*OJK",
            r"(?i)Dear\s*Helpdesk",
            r"(?i)Mohon\s*bantuannya",
            r"(?i)Terimakasih",
            r"(?i)Hormat\s*kami",
            r"(?i)Regard",
            r"(?i)Atas\s*perhatian\s*dan\s*kerja\s*samanya",
            r"(?i)Selamat\s*pagi",
            r"(?i)Dengan\s*hormat",
            r"(?i)Perhatian",
            r"(?i)Caution",
            r"(?i)Peringatan",
            r"(?i)Harap\s*diperhatikan",
            r"(?i)Terlampir",
            r"(?i)Mohon\s*kerjasamanya",
            r"(?i)Mohon\s*informasi",
            r"(?i)Sehubungan\s*dengan",
            r"(?i)Kepada",
            r"(?i)Kami\s*moho",
            r"(?i)Terkait",
            r"(?i)Berikut\s*kami\s*sampaikan",
            r"(?i)Jika\s*Anda\s*bukan\s*penerima\s*yang\s*dimaksud",
            r"(?i)Email\s*ini\s*hanya\s*ditujukan\s*untuk\s*penerima",
            r"(?i)Mohon\s*segera\s*memberitahukan\s*kami",
            r"(?i)Jika\s*Anda\s*memerima\s*ini\s*secara\s*tidak\s*seganja",
            r"(?i)Mohon\s*dihapus",
            r"(?i)Kami\s*mengucapkan\s*terima\s*kasi",
            r"(?i)Mohon\s*perhatian",
            r"(?i)From\s*.*",
            r"(?i)\n+",
            r"(?i)\s{2,}",
            r"(?i):\s*e-Mail\s*ini\s*termasuk\s∗seluruh\s∗lampirannya\s∗,\s∗bilang\s∗adatermasuk\s*seluruh\s*lampirannya\s*,\s*bilang\s*ada\s*hanya\s*ditujukan\s*penerima\s*yang\s*tercantum\s*di\s*atas.*",
            r"(?i):\s*This\s*electronic\s*mail\s*and\s*/\s*or\s*any\s*files\s*transmitted\s*with\s*it\s*may\s*contain\s*confidential\s*or\s*copyright\s*PT\.\s*Jasa\s*Raharja.*",
            r"(?i)PT\.\s*Jasa\s*Raharja\s*tidak\s*bertanggung\s*jawab\s*atas\s*kerugian\s*yang\s*ditimbulkan\s*oleh\s*virus\s*yang\s*ditularkan\s*melalui\s*e-Mail\s*ini.*",
            r"(?i)Jika\s*Anda\s*secara\s*tidak\s*seganja\s*menerima\s*e-Mail\s*ini\s*,\s*untuk\s*segera\s*memberitahukan\s*ke\s*alamat\s*e-Mail\s*pengirim\s*serta\s*menghapus\s*e-Mail\s*ini\s*beserta\s*seluruh\s*lampirannya\s*.*",
            r"(?i)\s*Please\s*reply\s*to\s*this\s*electronic\s*mail\s*to\s*notify\s*the\s*sender\s*of\s*its\s*incorrect\s*delivery\s*,\s*and\s*then\s*delete\s*both\s*it\s*and\s*your\s*reply.*",
            ]
        for pat in pats:
            t = re.sub(pat, "", t, flags=re.IGNORECASE)
        return t

    # Cut off footer/forward
    def cut_off_general(c):
        cut_keywords = [
           "PT Mandiri Utama FinanceSent: Wednesday, November 6, 2024 9:11 AMTo",
            "Atasdan kerjasama, kami ucapkan h Biro Hukum dan KepatuhanPT Jasa Raharja (Persero)Jl. HR Rasuna Said Kav. C-2 12920Jakarta Selatan",
            "h._________",
            "h Imawan FPT ABC Multifinance Pesan File Kirim (PAPUPPK/2024-12-31/Rutin/Gagal)Kotak MasukTelusuri semua pesan berlabel Kotak MasukHapus",
            "KamiBapak/Ibu untuk pencerahannya",
            "kami ucapkan h. ,",
            "-- , DANA PENSIUN BPD JAWA",
            "sDian PENYANGKALAN.",
            "------------------------Dari: Adrian",
            "hormat saya RidwanForwarded",
            "--h, DANA PENSIUN WIJAYA",
            "Mohon InfonyahKantor",
            "an arahannya dari Bapak/ Ibu",
            "ya untuk di check ya.Thank",
            "Kendala:Thank youAddelin",
            ",Sekretaris DAPENUrusan Humas & ProtokolTazkya",
            "Mohon arahannya.Berikut screenshot",
            "Struktur_Data_Pelapor_IJK_(PKAP_EKAP)_-_Final_2024",
            "Annie Clara DesiantyComplianceIndonesia",
            "Dian Rosmawati RambeCompliance",
            "Beararti apakah,Tri WahyuniCompliance DeptPT.",
            "Dengan alamat email yang didaftarkan",
            "dan arahan",
            ",AJB Bumiputera",
            "’h sebelumnya Afriyanty",
            "PENYANGKALAN.",
            "h Dana Pensiun PKT",
            ", h , Tasya PT.",
            "Contoh: 10.00",
            "hAnnisa Adelya SerawaiPT Fazz",
            "sebagaimana gambar di bawah ini",
            "PT Asuransi Jiwa SeaInsure On Fri",
            "hJana MaesiatiBanking ReportFinance",
            "Tembusan",
            "Sebagai referensi",
            "hAdriansyah",
            "h atas bantuannya Dwi Anggina",
            "PT Asuransi Jiwa SeaInsure",
            "dengan notifikasi dibawah ini",
            "Terima ksh",
            ": DISCLAIMER",
            "Sebagai informasi",
            "nya. h.Kind s,Melati",
            ": DISCLAIMER",
            "Petugas AROPT",
            "h,Julianto",
            "h,Hernawati",
            "Dana Pensiun Syariah",
            ",Tria NoviatyStrategic"
        ]
        for kw in cut_keywords:
            if kw in c:
                c = c.split(kw)[0]
        return c

    t = remove_dear_ojk(text)
    comp = extract_complaint(t)
    comp = clean_email_text(comp)
    comp = cut_off_general(comp)
    return comp

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
    return output.getvalue()

# --- Streamlit UI ---
st.title("📊 Aplikasi Klasifikasi Kendala SIPEDULI")
st.markdown("""
1. Upload file CSV/Excel  
2. Pilih sheet (jika Excel)  
3. Pilih kolom teks kendala  
4. Klik **Proses Prediksi**  
5. Download hasil  
""")
st.sidebar.header("📁 Upload File")
file = st.sidebar.file_uploader("📎 Pilih file CSV/Excel", type=['csv','xlsx'])

if file:
    try:
        if file.name.endswith('xlsx'):
            xls = pd.ExcelFile(file)
            sheet = st.sidebar.selectbox("Pilih Sheet", xls.sheet_names)
            data = pd.read_excel(xls, sheet_name=sheet)
        else:
            data = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        st.stop()

    st.subheader("👀 Data Preview")
    st.dataframe(data.head())

    col = st.selectbox("📝 Kolom teks kendala ada di kolom:", data.columns)

    if st.button("🚀 Proses Prediksi"):
        # 0) Simpan salinan original
        original = data.copy()
        # 1) Buat df khusus kolom terpilih
        df_sel = data[[col]].copy()

        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner("⏳ Memproses......"):
            status_text.text("1/5: Preprocessing teks...")
            # Preprocessing dasar dulu
            df_sel['Pre_Cleaned'] = df_sel[col].fillna('').apply(preprocess_text)
            
            # Ambil unique teks hasil preprocess
            unique_texts = df_sel['Pre_Cleaned'].unique().tolist()
            
            # Dapatkan mapping kata dari cache statis + stemming runtime
            stemmed_map = get_stemmed_mapping(unique_texts)  # ini versi cache statis + stemming
            
            # Map hasil stemmed ke kolom Cleaned
            df_sel['Cleaned'] = df_sel['Pre_Cleaned'].map(stemmed_map)
            
            progress_bar.progress(20)


            # 2/5 TF-IDF
            status_text.text("2/5: Transformasi TF-IDF...")
            valid_mask = df_sel['Cleaned'].notna()
            tfidf_m = tfidf.transform(df_sel.loc[valid_mask, 'Cleaned'].tolist())
            progress_bar.progress(40)

            # 3/5 SVD
            status_text.text("3/5: Reduksi dimensi dengan SVD...")
            svd_f = svd.transform(tfidf_m)
            progress_bar.progress(60)

            # 4/5 Prediksi
            status_text.text("4/5: Memprediksi label...")
            df_sel.loc[valid_mask, 'Label'] = model.predict(svd_f)
            progress_bar.progress(80)

            # 5/5 Mapping Topik
            status_text.text("5/5: Mapping ke topik...")
            label_to_topik = {
                0: 'Bukti tanda terima penyampaian laporan pelaksanaan LIK',
                1: 'Kendala jumlah kegiatan masih 0',
                2: 'Kendala login SIPEDULI (user terblokir)',
                3: 'Kendala penyampaian LLP',
                4: 'Kendala penyampaian LSA',
                5: 'Kendala penyampaian laporan pelaksanaan LIK',
                6: 'Kendala penyampaian laporan rencana LIK',
                7: 'Kendala upload dokumentasi pada laporan pelaksanaan LIK',
                8: 'Konfirmasi denda keterlambatan',
                9: 'Lainnya',
               10: 'Menambah kegiatan di laporan rencana LIK',
               11: 'Permintaan perubahan password',
               12: 'Permohonan koreksi laporan LIK',
               13: 'Perubahan data pada aplikasi SIPEDULI'
            }
            df_sel['Prediksi_Topik'] = df_sel['Label'].map(label_to_topik)
            progress_bar.progress(100)

        # 2) Tempelkan hasil kembali ke original
        original = original.join(df_sel[['Prediksi_Topik']])

        st.success("✅ Selesai")
        st.dataframe(original)

        # Download hasil lengkap
        excel_data = to_excel(original)
        st.download_button(
            label="📥 Download Hasil Lengkap",
            data=excel_data,
            file_name='hasil_lengkap_sipeduli.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
else:
    st.info("Silakan upload file terlebih dahulu.")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>Developed by Mesakh Besta Anugrah • OJK Internship 2025</div>",
    unsafe_allow_html=True
)

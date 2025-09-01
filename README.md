# 🍅 Pomidor Tasnifi – YOLOv8 yordamida

Ushbu loyiha pomidorlarning **pishgan (ripe)** va **xom (unripe)** holatini aniqlash uchun **YOLOv8** modelidan foydalanadi. Loyiha dastlab **Google Colab** uchun yozilgan, lekin **Visual Studio Code** yoki **GitHub Codespaces** muhitida ham ishlashga moslashtirilgan.  

---

## 📂 Loyihaning Tuzilishi

```
pomidor/
│── model_tayyorlash.py      # Asosiy Python kodi (EDA + dataset tayyorlash + YOLO config)
│── rasmlar/             # Dataset (Images/ va labels/)
│   ├── Images/          # .jpeg rasmlar
│   └── labels/          # YOLO formatdagi .txt annotatsiyalar
│── yolo_data/           # Train/test split uchun avtomatik yaratiladigan papka
│── runs/                # YOLOv8 o‘qitish natijalari (avtomatik yaratiladi)
│── .gitignore           # Git uchun kerakli sozlamalar
│── README.md            # Ushbu fayl
```

---

## ⚙️ O‘rnatish

1. Reponi yuklab oling:

```bash
git clone https://github.com/username/pomidor.git
cd pomidor
```

2. Virtual environment yarating va aktivlashtiring:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Kerakli kutubxonalarni o‘rnating:

```bash
pip install -r requirements.txt
```

Agar `requirements.txt` bo‘lmasa, asosiy kutubxonalar:  
```bash
pip install ultralytics scikit-learn matplotlib opencv-python pillow numpy
```

---

## 📊 Dataset

- `rasmlar/Images/` – `.jpeg` formatdagi pomidor rasmlari
- `rasmlar/labels/` – YOLOv8 formatdagi `.txt` annotatsiyalar  
  (sinflar: **0 = unripe**, **1 = ripe**)

---

## 🔎 EDA (Exploratory Data Analysis)

Kod avtomatik ravishda:
- O‘rtacha rasm o‘lchamlarini hisoblaydi
- Sinflar taqsimotini chiqaradi
- Ba’zi namunaviy rasmlarni ko‘rsatadi
- RGB histogramlarini chizadi

Namuna natijalar:  
```
Average Image Width: 1374.88 pixels
Average Image Height: 991.36 pixels
ripe: 429
unripe: 440
Number of ripe images: 91
Number of unripe images: 86
```

---

## 🏋️‍♂️ Modelni O‘qitish

Modelni o‘qitish `kutubxonalar.py` ichida yozilgan.  
Siz kodni ishga tushirganingizda dataset `train/test` ga bo‘linadi va `dataset.yaml` avtomatik yaratiladi.

YOLOv8 o‘qitish qismi:

```python
from ultralytics import YOLO

# Oldindan o‘qitilgan YOLOv8 modelini yuklash
model = YOLO('yolov8n.pt')

# Dataset yo‘li
data_yaml_path = "yolo_data/dataset.yaml"

# Modelni o‘qitish
results = model.train(
    data=data_yaml_path,
    epochs=130,
    imgsz=720,
    device="cpu",   # GPU bo‘lsa 'cuda'
    lr0=0.01,
    lrf=0.001,
    save_period=10
)

# Modelni saqlash
model.save("pomidor_model.pt")
```

---

## 🧪 Modelni Test Qilish

O‘qitilgan modelni test qilish uchun:

```python
# Modelni yuklash
model = YOLO("pomidor_model.pt")

# Test rasmlarida sinov qilish
results = model.predict("yolo_data/test/images", conf=0.5, iou=0.6)

# Natijalarni ko‘rish
for r in results:
    r.show()   # vizual natija
    r.save("natijalar/")  # saqlash
```

---

## 🚀 GitHub Codespaces bo‘yicha maslahatlar

- `epochs` sonini kamaytirish (`20–50`) – resursni tejash uchun  
- `device="cpu"` ishlatish (agar GPU yo‘q bo‘lsa)  
- Agar trenirovka davomida **Terminated** chiqsa:
  - `imgsz` ni kichraytirish (640 yoki 512)
  - `batch=4` qo‘shish

---

## 📌 Natija

Ushbu loyiha yordamida siz pomidorlarni **pishgan / pishmagan** deb avtomatik tasniflay oladigan **YOLOv8 modeli** yaratishingiz mumkin.  
O‘qitilgan model `pomidor_model.pt` faylida saqlanadi va keyinchalik real hayotiy dasturlarda ishlatilishi mumkin.  

---

✍️ Muallif: *[Sizning ismingiz]*  
📅 Sana: *2025*  

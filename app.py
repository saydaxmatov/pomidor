from ultralytics import YOLO

# O‘qitilgan modelni yuklash
model = YOLO("runs/detect/train/weights/best.pt")

# Sinov
results = model.predict("pomidor.jpg", conf=0.5, iou=0.6)
results[0].show()

print("#"*20)
print(results)

# Buat pakai test library yang sudah terinstall atau belum
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    print("Detectron2 sudah terinstal.")
except ImportError as e:
    print(f"Error: {e}")
au
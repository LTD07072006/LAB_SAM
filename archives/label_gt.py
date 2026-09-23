import os
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    sample_dir = "./fire-samples"
    image_paths = glob.glob(os.path.join(sample_dir, "*.*"))
    
    ground_truth_pixels = {}
    
    print("[*] CÔNG CỤ TẠO GROUND TRUTH")
    print(" - Hãy dùng chuột TRÁI click vào điểm ĐÁY NGỌN LỬA (nơi chạm đất).")
    print("-" * 50)
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        try:
            img = mpimg.imread(img_path)
        except Exception as e:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        ax.set_title(f"Click vào ĐÁY NGỌN LỬA: {img_name}")
        
        # Lấy tọa độ 1 lần click chuột
        pts = plt.ginput(1, timeout=0) 
        plt.close()
        
        if pts:
            u, v = pts[0]
            ground_truth_pixels[img_name] = [int(u), int(v)]
            print(f"[+] Đã lưu {img_name}: u={int(u)}, v={int(v)}")
        else:
            print(f"[-] Bỏ qua {img_name}")

    # Xuất ra file JSON để benchmark.py đọc
    with open("ground_truth.json", "w") as f:
        json.dump(ground_truth_pixels, f, indent=4)
    print("\n[*] HOÀN TẤT! Đã lưu file 'ground_truth.json'.")

if __name__ == "__main__":
    main()
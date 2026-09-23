import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from fire_detector import FireDetector
from locator import CameraGeometry, GridMap, intersect_ray_with_grid

def main():
    # 1. KHỞI TẠO HỆ THỐNG
    print("[*] Khởi tạo hệ thống AI và Định vị 3D...")
    # Cập nhật đường dẫn file best.pth nếu cần
    detector = FireDetector(model_path="./fire-model-data/best.pth") 
    grid_map = GridMap()
    
    # Giả lập ma trận Camera
    K = [[800, 0, 640], [0, 800, 360], [0, 0, 1]]
    theta = np.radians(45)
    R = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    C_world = np.array([0, -20, 25]) 
    t = -np.array(R) @ C_world.reshape(3, 1)
    camera = CameraGeometry(K, R, t)

    # 2. KHỞI TẠO BIỂU ĐỒ 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vẽ bề mặt địa hình Grid-Map
    X_grid = np.linspace(-30, 30, 100)
    Y_grid = np.linspace(-30, 30, 100)
    X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
    Z_mesh = grid_map.get_elevation(X_mesh, Y_mesh)
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='terrain', alpha=0.5, edgecolor='none')
    
    # Vẽ Camera CCTV
    ax.scatter(*C_world, color='blue', s=150, marker='s', label='CCTV Camera')

    # 3. QUÉT TOÀN BỘ ẢNH TRONG THƯ MỤC FIRE-SAMPLES
    sample_dir = "./fire-samples"
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    
    if not os.path.exists(sample_dir):
        print(f"[!] Không tìm thấy thư mục: {sample_dir}")
        return

    # Lấy danh sách đường dẫn tất cả các ảnh
    image_paths = [p for p in glob.glob(os.path.join(sample_dir, "*.*")) if p.lower().endswith(valid_exts)]
    print(f"[*] Tìm thấy {len(image_paths)} ảnh trong thư mục {sample_dir}.")

    # Bảng màu ngẫu nhiên để phân biệt các tia chiếu
    colors = plt.get_cmap('tab10', len(image_paths))

    # 4. VÒNG LẶP XỬ LÝ TỪNG ẢNH
    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        print(f"\n--- Đang xử lý: {img_name} ---")
        
        # Nhận diện 2D
        p_fire, u, v = detector.detect(img_path)
        print(f"  > Xác suất lửa: {p_fire*100:.1f}% | Tọa độ ảnh: (u={u}, v={v})")
        
        if p_fire < 0.5:
            print("  > [Bỏ qua] Không phát hiện lửa.")
            continue

        # Định vị 3D
        C, ray_dir = camera.pixel_to_ray(u, v)
        fire_3d_pos = intersect_ray_with_grid(C, ray_dir, grid_map)

        if fire_3d_pos is None:
            print("  > [Cảnh báo] Tia chiếu bay ra ngoài không gian, không chạm đất.")
            continue
            
        print(f"  > Tọa độ 3D: X={fire_3d_pos[0]:.2f}, Y={fire_3d_pos[1]:.2f}, Z={fire_3d_pos[2]:.2f}")

        # Trực quan hóa tia và điểm của bức ảnh này lên biểu đồ chung
        color = colors(i)
        
        # Vẽ tia chiếu
        ax.plot([C_world[0], fire_3d_pos[0]], 
                [C_world[1], fire_3d_pos[1]], 
                [C_world[2], fire_3d_pos[2]], 
                color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Vẽ điểm lửa 3D
        ax.scatter(*fire_3d_pos, color=color, s=100, marker='*', edgecolor='black')
        
        # Gắn thẻ tên ảnh tại vị trí cháy để dễ đối chiếu
        ax.text(fire_3d_pos[0], fire_3d_pos[1], fire_3d_pos[2] + 2, 
                f"Img {i+1}", color='black', fontsize=8, weight='bold')

    # Hoàn thiện biểu đồ
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Elevation Z (m)')
    ax.set_title('Mô phỏng Định vị 3D cho 10 mẫu ảnh đám cháy')
    
    # Chỉ hiển thị nhãn Camera 1 lần, tránh lặp lại
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    print("\n[*] Đã xử lý xong toàn bộ ảnh! Đang hiển thị kết quả 3D...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
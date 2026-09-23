import numpy as np
import trimesh
import time

def main():
    print("[INIT] Nạp bản đồ lưới...")
    mesh = trimesh.load('Office_1_mesh.ply')
    intersector = mesh.ray
    
    # Ép màu xám tối
    mesh.visual.face_colors = [120, 120, 120, 255]
    
    # Lấy giới hạn không gian của toàn bộ căn phòng (Min, Max theo X, Y, Z)
    min_b, max_b = mesh.bounds
    
    # 1. TÍNH TOÁN TỌA ĐỘ BÊN TRONG PHÒNG
    # Camera treo ở góc cao (thụt vào 10% so với trần và tường để nằm HẲN BÊN TRONG)
    cam_pos = np.array([
        min_b[0] + (max_b[0] - min_b[0]) * 0.1,
        max_b[1] - (max_b[1] - min_b[1]) * 0.1,
        max_b[2] - (max_b[2] - min_b[2]) * 0.1 
    ])
    
    # Giả định ngọn lửa nằm ở dưới sàn (cách đáy 2% để không bị lọt gầm)
    ground_truth = np.array([
        (min_b[0] + max_b[0]) * 0.6, 
        (min_b[1] + max_b[1]) * 0.4, 
        min_b[2] + (max_b[2] - min_b[2]) * 0.02
    ])
    
    perfect_ray_dir = (ground_truth - cam_pos)
    perfect_ray_dir = perfect_ray_dir / np.linalg.norm(perfect_ray_dir)

    # Ép nạp cache BVH
    _ = intersector.intersects_location(
        ray_origins=cam_pos.reshape(1, 3), 
        ray_directions=perfect_ray_dir.reshape(1, 3)
    )

    # 2. ĐÂM TIA VÀ ĐO THỜI GIAN
    start_time = time.perf_counter()
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=cam_pos.reshape(1, 3), 
        ray_directions=perfect_ray_dir.reshape(1, 3)
    )
    end_time = time.perf_counter()
    
    latency = (end_time - start_time) * 1000
    
    if len(locations) > 0:
        hit_point = locations[0]
        
        # 3. TRỰC QUAN HÓA
        scene = trimesh.Scene([mesh])
        
        # Tia ngắm: Màu Vàng
        ray_path = trimesh.load_path(np.array([cam_pos, hit_point]))
        ray_path.colors = np.array([[255, 255, 0, 255]])
        scene.add_geometry(ray_path)
        
        # Lửa: Màu Đỏ
        hit_sphere = trimesh.creation.icosphere(radius=0.15)
        hit_sphere.apply_translation(hit_point)
        hit_sphere.visual.face_colors = [255, 0, 0, 255]
        scene.add_geometry(hit_sphere)
        
        # Camera: Màu Xanh Lá
        cam_sphere = trimesh.creation.icosphere(radius=0.25)
        cam_sphere.apply_translation(cam_pos)
        cam_sphere.visual.face_colors = [0, 255, 0, 255]
        scene.add_geometry(cam_sphere)
        
        print("\n[MẸO QUAN TRỌNG]")
        print("-> Nhấn phím 'w' để bật lưới khung dây!")
        
        scene.show(caption=f"Execution Time: {latency:.4f} ms")
    else:
        print("Tia ngắm bị trượt!")

if __name__ == "__main__":
    main()
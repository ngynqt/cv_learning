import cv2
import torch
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# ==========================================
# CẤU HÌNH VÀ KHỞI TẠO MÔ HÌNH
# ==========================================

# Thiết lập device: Dùng GPU nếu có, ngược lại dùng CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}")

# Khởi tạo MTCNN để phát hiện khuôn mặt
# mtcnn_realtime: dùng cho video, detect nhiều người (keep_all=True)
mtcnn_realtime = MTCNN(keep_all=True, device=device)
# mtcnn_db: dùng để trích xuất 1 khuôn mặt chuẩn từ ảnh database (keep_all=False)
mtcnn_db = MTCNN(keep_all=False, device=device)

# Khởi tạo FaceNet (InceptionResnetV1) để trích xuất embedding
# Pretrained trên tập VGGFace2
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# ==========================================
# CÁC HÀM TIỆN ÍCH CHÍNH
# ==========================================

def get_embedding(face_tensor):
    """
    Trích xuất vector đặc trưng (embedding) từ một face tensor sử dụng FaceNet.
    Input: face_tensor dạng (1, 3, 160, 160)
    Output: mảng numpy 1D các đặc trưng (embedding)
    """
    with torch.no_grad(): # Không cần tính gradient khi inference
        embedding = resnet(face_tensor).cpu().numpy()
    return embedding

def compare_faces(current_emb, db_embeddings, threshold=0.7):
    """
    So sánh embedding hiện tại với các embedding trong database.
    Tính toán Cosine Similarity và kiểm tra với threshold.
    """
    best_match_name = "Unknown"
    highest_sim = -1.0
    
    # Duyệt qua từng người trong database
    for name, db_emb in db_embeddings.items():
        # cosine_similarity trả về ma trận, lấy giá trị [0][0]
        sim = cosine_similarity(current_emb, db_emb)[0][0]
        
        if sim > highest_sim:
            highest_sim = sim
            if sim > threshold:
                best_match_name = name
                
    return best_match_name, highest_sim

def load_database(db_path="face_database"):
    """
    Load các ảnh mẫu (database), trích xuất khuôn mặt và lưu các embeddings.
    """
    db_embeddings = {}
    
    # Tạo folder nếu chưa có
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        print(f"[INFO] Đã tạo thư mục database tại '{db_path}'.")
        print("[INFO] Bạn có thể chép ảnh vào đây (VD: Tuong.jpg) để nhận diện.")
        return db_embeddings

    print("\n[INFO] Đang tải database khuôn mặt...")
    for filename in os.listdir(db_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(db_path, filename)
            
            try:
                # MTCNN yêu cầu ảnh ở định dạng PIL RGB
                img = Image.open(img_path).convert('RGB')
                
                # Phát hiện và crop khuôn mặt, trả về tensor
                face_tensor = mtcnn_db(img)
                
                if face_tensor is not None:
                    # Thêm chiều batch_size = 1 -> tensor có dạng (1, 3, 160, 160)
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    emb = get_embedding(face_tensor)
                    db_embeddings[name] = emb
                    print(f"  + Thêm '{name}' thành công.")
                else:
                    print(f"  - Cảnh báo: Không tìm thấy khuôn mặt trong {filename}")
            except Exception as e:
                print(f"  - Lỗi khi đọc ảnh {filename}: {e}")
                
    print(f"[INFO] Hoàn tất tải database với {len(db_embeddings)} người.\n")
    return db_embeddings


# ==========================================
# CHƯƠNG TRÌNH CHÍNH (MAIN PROCESS)
# ==========================================

def detect_and_recognize():
    # 1. Load database các khuôn mặt đã biết
    db_embeddings = load_database()
    
    # 2. Khởi tạo Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Không thể truy cập Camera. Vui lòng kiểm tra lại thiết bị!")
        return
        
    print("[INFO] Webcam đang chạy...")
    print("[INFO] Hướng dẫn:")
    print(" - Nhấn 'q' để Thoát.")
    print(" - Nhấn 's' để Lưu người mới (Chụp ảnh và thêm vào database).")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Không thể đọc frame từ Camera.")
            break
            
        # OpenCV mặc định dùng BGR, chuyển sang RGB cho hệ sinh thái PIL/PyTorch
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        try:
            # Detect bounds (boxes) và khuôn mặt cắt sẵn (face_tensors)
            boxes, probs = mtcnn_realtime.detect(img_pil)
            face_tensors = mtcnn_realtime(img_pil)
            
            # Nếu phát hiện thấy khuôn mặt
            if boxes is not None and face_tensors is not None:
                for i, box in enumerate(boxes):
                    # Lấy tọa độ bounding box
                    x1, y1, x2, y2 = [int(b) for b in box]
                    
                    # Lấy tensor khuôn mặt thứ i và chuyển sang GPU/CPU
                    target_tensor = face_tensors[i].unsqueeze(0).to(device)
                    
                    # 3. Trích xuất embedding bằng FaceNet
                    current_emb = get_embedding(target_tensor)
                    
                    # 4. So sánh với Database
                    name, sim = compare_faces(current_emb, db_embeddings, threshold=0.7)
                    
                    # Định dạng hiển thị
                    if name != "Unknown":
                        label = f"Matched: {name} ({sim:.2f})"
                        color = (0, 255, 0) # Màu xanh lục
                    else:
                        label = f"Unknown ({sim:.2f})"
                        color = (0, 0, 255) # Màu đỏ
                        
                    # Vẽ Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Vẽ nền chữ cho dễ đọc
                    cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, cv2.FILLED)
                    
                    # Chèn Label lên trên frame
                    cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            else:
                # Nếu không có khuôn mặt (yêu cầu hiển thị khi không có)
                cv2.putText(frame, "No face detected", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
        except Exception as e:
            pass # Bỏ qua các lỗi tracking khung hình nhỏ của MTCNN
            
        # Hiển thị frame
        cv2.imshow('Realtime Face Recognition', frame)
        
        # Bắt sự kiện phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        # [Bonus] - Nhấn phím 's' để chụp hình người đứng trước camera và thêm vào database
        elif key == ord('s'):
            count = len(db_embeddings) + 1
            new_name = f"Person_{count}"
            save_path = os.path.join("face_database", f"{new_name}.jpg")
            
            # Lưu ảnh
            cv2.imwrite(save_path, frame)
            print(f"[INFO] Đã thêm người mới: '{new_name}' vào mục 'face_database/'")
            
            # Reload lại database ngay lập tức
            db_embeddings = load_database()
            print("[INFO] Đã update lại Database sau khi thêm!")

    # Giải phóng resource
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_recognize()
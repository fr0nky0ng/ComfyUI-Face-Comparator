import insightface
import cv2
import numpy as np
import torch

class FaceComparator:
    """
    一個 ComfyUI 節點，用於比較兩張圖片中的人臉是否為同一個人。
    """
    
    # --- 1. 初始化 InsightFace 模型 ---
    def __init__(self):
        # 將模型加載放在構造函數中，確保只在啟動時加載一次
        print("InsightFace: Initializing FaceAnalysis model...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l', root='~/.insightface', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        print("InsightFace: Model prepared.")

    # --- 2. 設定 ComfyUI 節點的元數據 ---
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face1": ("IMAGE",),
                "face2": ("IMAGE",),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.65, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display": "slider" 
                }),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("text", "is_same_person")
    FUNCTION = "compare"
    CATEGORY = "Face"

    # --- 3. 核心功能函數 ---
    def compare(self, image1: torch.Tensor, image2: torch.Tensor, similarity_threshold: float):
        """
        比較兩張輸入圖片中的人臉。
        """
        # 將 ComfyUI 的 Tensor 格式轉換為 OpenCV 的 NumPy 格式
        # Tensor: [Batch, Height, Width, Channel], 0-1 float
        # NumPy: [Height, Width, Channel], 0-255 uint8, BGR color
        img1_np = (image1.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img2_np = (image2.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        
        # 從 RGB 轉換為 BGR，因為 OpenCV 默認使用 BGR
        img1_bgr = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)

        # 檢測人臉並提取特徵
        faces1 = self.app.get(img1_bgr)
        faces2 = self.app.get(img2_bgr)

        # 檢查是否檢測到人臉
        if not faces1:
            return ("Error: No faces detected in image 1.", False)
        if not faces2:
            return ("Error: No faces detected in image 2.", False)

        # 提取第一個檢測到的人臉的特徵
        face1 = faces1[0]
        face2 = faces2[0]
        embedding1 = face1.normed_embedding
        embedding2 = face2.normed_embedding
        
        # 計算餘弦相似度
        similarity = np.dot(embedding1, embedding2)

        # 比較相似度與閾值，並生成結果
        if similarity > similarity_threshold:
            is_same = True
            result_text = f"Result: The same person.\n(Similarity: {similarity:.4f} > Threshold: {similarity_threshold})"
        else:
            is_same = False
            result_text = f"Result: Not the same person.\n(Similarity: {similarity:.4f} <= Threshold: {similarity_threshold})"

        return (result_text, is_same)

# --- 4. ComfyUI 節點註冊 ---
# 這是 ComfyUI 加載自定義節點所需的標準映射
NODE_CLASS_MAPPINGS = {
    "FaceComparator": FaceComparator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceComparator": "Face Comparator"
}

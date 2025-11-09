import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import os
import warnings
import sys

# Add the project root to path to import pointpillars module
sys.path.append('/home/yenhsiu/eira_git/PointPillars_split_inference')

from pointpillars.model import PointPillars
from pointpillars.dataset import Kitti, get_dataloader

warnings.filterwarnings('ignore')

PYTORCH_WEIGHTS_PATH = '/home/yenhsiu/eira_git/PointPillars_split_inference/pretrained/epoch_160.pth'
ONNX_MODEL_PATH = "pointpillars.onnx"
TF_SAVED_MODEL_PATH = "saved_model"

# PointPillars Configuration
NCLASSES = 3  # Car, Pedestrian, Cyclist
VOXEL_SIZE = [0.16, 0.16, 4]
POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 1]
MAX_NUM_POINTS = 32
MAX_VOXELS = (16000, 40000)

TFLITE_MODEL = "pointpillars.tflite"
DATA_DIR = "/home/yenhsiu/datasets"


def representative_dataset_gen():
    """
    生成器函式，從KITTI資料集讀取點雲數據，
    經過預處理後，轉換成 TensorFlow Lite 轉換器所需的格式和型別。
    """
    print("初始化代表性資料集生成器...")
    
    # 使用KITTI數據集的驗證集來產生代表性數據
    val_dataset = Kitti(data_root=DATA_DIR, split='val')
    val_dataloader = get_dataloader(dataset=val_dataset, batch_size=1, num_workers=4, shuffle=False)
    
    # 限制只使用少量點雲作為代表性資料集，加快量化過程
    count = 0
    for data_dict in val_dataloader:
        if count >= 20:  # 使用20個點雲樣本
            break
            
        batched_pts = data_dict['batched_pts']
        # 使用模型的pillar_layer處理點雲，獲取輸入張量
        with torch.no_grad():
            pillars, coors_batch, npoints_per_pillar = model.pillar_layer(batched_pts)
            pillars_np = pillars.cpu().numpy().astype(np.float32)
            coors_batch_np = coors_batch.cpu().numpy().astype(np.int32)
            npoints_np = npoints_per_pillar.cpu().numpy().astype(np.int32)
            
        # 將處理後的點雲數據作為代表性數據提供給TFLite轉換器
        yield [pillars_np, coors_batch_np, npoints_np]
        
        count += 1
        if count % 5 == 0:
            print(f"  提供了 {count} 個點雲樣本...")
    
    print("代表性資料集提供完畢。")

print("=== 步驟 1: 從 PyTorch 載入 PointPillars 模型並轉換為 ONNX ===")

# 初始化PointPillars模型
model = PointPillars(nclasses=NCLASSES, 
                    voxel_size=VOXEL_SIZE, 
                    point_cloud_range=POINT_CLOUD_RANGE,
                    max_num_points=MAX_NUM_POINTS, 
                    max_voxels=MAX_VOXELS)

# 載入預訓練權重
if os.path.exists(PYTORCH_WEIGHTS_PATH):
    checkpoint = torch.load(PYTORCH_WEIGHTS_PATH, map_location='cpu')
    # 根據checkpoint的儲存方式調整載入方式
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"成功從 '{PYTORCH_WEIGHTS_PATH}' 載入權重。")
else:
    print(f"找不到權重文件: '{PYTORCH_WEIGHTS_PATH}'，使用隨機初始化權重。")

# 設置為評估模式
model.eval()

# 建立一個模擬的點雲輸入，使用更真實的大小
print("創建真實大小的 dummy 輸入...")
dummy_pts = torch.randn(15000, 4)  # 15000個點，更接近真實點雲大小
dummy_pillars, dummy_coors_batch, dummy_npoints = model.pillar_layer([dummy_pts])

print(f"Dummy data 形狀:")
print(f"  pillars: {dummy_pillars.shape}")
print(f"  coors_batch: {dummy_coors_batch.shape}")
print(f"  npoints: {dummy_npoints.shape}")

# 為PointPillars模型創建一個前向傳播函數，該函數僅包括推理部分
class PointPillarsInference(torch.nn.Module):
    def __init__(self, model):
        super(PointPillarsInference, self).__init__()
        self.model = model
        
    def forward(self, pillars, coors_batch, npoints_per_pillar):
        # 只使用前向傳播的推理部分，不包括訓練相關的計算
        features = self.model.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        x = self.model.backbone(features)
        x = self.model.neck(x)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.model.head(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred

# 創建推理模型
inference_model = PointPillarsInference(model)
inference_model.eval()

# 導出ONNX模型
torch.onnx.export(
    inference_model,
    (dummy_pillars, dummy_coors_batch, dummy_npoints),
    ONNX_MODEL_PATH,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['pillars', 'coors_batch', 'npoints_per_pillar'],
    output_names=['bbox_cls_pred', 'bbox_pred', 'bbox_dir_cls_pred'],
    dynamic_axes={
        'pillars': {0: 'num_pillars'},
        'coors_batch': {0: 'num_pillars'},
        'npoints_per_pillar': {0: 'num_pillars'},
        'bbox_cls_pred': {0: 'batch_size'},
        'bbox_pred': {0: 'batch_size'},
        'bbox_dir_cls_pred': {0: 'batch_size'}
    }
)

print(f"模型已成功轉換並儲存至 '{ONNX_MODEL_PATH}'")
print("-" * 50)

print("=== 步驟 2: 從 ONNX 轉換為 TensorFlow SavedModel ===")

try:
    # 載入 ONNX 並轉為 TensorFlow
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(TF_SAVED_MODEL_PATH)

    print(f"TensorFlow SavedModel 已儲存至 '{TF_SAVED_MODEL_PATH}' 資料夾。")
except Exception as e:
    print(f"ONNX轉換到TensorFlow時出錯: {e}")
    print("請確認已安裝onnx-tf，並且ONNX模型兼容TensorFlow轉換器。")
    sys.exit(1)

print("-" * 50)

# -------------------------------
# TFLite Converter 設定
# -------------------------------
print("開始設定 TFLite 轉換器...")
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_PATH)
    
    # 設定優化選項
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 啟用 TF Select 以支援不相容的操作（如 tf.Range, tf.Conv2D）
    # 這允許模型在 TFLite 運行時使用 TensorFlow 操作的回退
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # 使用 TFLite 內建操作
        tf.lite.OpsSet.SELECT_TF_OPS     # 啟用 TF Select 回退
    ]
    
    # 嘗試進行INT8量化
    print("嘗試進行INT8量化（使用 TF Select）...")
    converter.representative_dataset = representative_dataset_gen
    
    # 允許在量化過程中使用 TF 操作
    converter.allow_custom_ops = True
    
    # 對於有TF Select的量化，使用混合模式
    converter.target_spec.supported_types = [tf.float16]
    
    # 對於浮點數運算，保持輸入和輸出為浮點數
    # 這樣在某些硬件上可能會有更好的性能和準確性
    print("開始轉換模型... 這個過程可能需要幾分鐘。")
    tflite_model = converter.convert()
    
    # 儲存量化模型
    with open(TFLITE_MODEL, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite 模型生成完成: {TFLITE_MODEL}")
    
    # 嘗試創建浮點模型（使用 TF Select）
    print("嘗試創建浮點模型（使用 TF Select）...")
    float_converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_PATH)
    float_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    float_converter.allow_custom_ops = True
    float_tflite_model = float_converter.convert()
    
    # 儲存浮點模型
    float_model_path = "pointpillars_float.tflite"
    with open(float_model_path, 'wb') as f:
        f.write(float_tflite_model)
    
    print(f"✅ 浮點 TFLite 模型生成完成: {float_model_path}")

except Exception as e:
    print(f"❌ 模型轉換失敗：{e}")
    print("請檢查 TF_SAVED_MODEL 路徑、代表性資料集或TensorFlow版本兼容性。")
    
    # 在出錯時嘗試使用更基本的設定
    print("嘗試使用基本設定重新轉換（啟用 TF Select）...")
    try:
        basic_converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_PATH)
        # 啟用 TF Select 支援
        basic_converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        basic_converter.allow_custom_ops = True
        
        tflite_model = basic_converter.convert()
        
        with open("pointpillars_basic.tflite", 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ 基本 TFLite 模型生成完成: pointpillars_basic.tflite")
    except Exception as e2:
        print(f"❌ 基本模型轉換也失敗：{e2}")
        print("請確保TensorFlow安裝正確，並嘗試更新到較新的版本。")
        
        # 最後嘗試：純浮點模型，不進行任何優化
        print("最後嘗試：創建純浮點模型，不進行優化...")
        try:
            minimal_converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_PATH)
            minimal_converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
            minimal_converter.allow_custom_ops = True
            # 不設置任何優化
            minimal_tflite_model = minimal_converter.convert()
            
            with open("pointpillars_minimal.tflite", 'wb') as f:
                f.write(minimal_tflite_model)
            
            print(f"✅ 最小化 TFLite 模型生成完成: pointpillars_minimal.tflite")
        except Exception as e3:
            print(f"❌ 所有轉換方法都失敗了：{e3}")
            sys.exit(1)

# 輸出文件大小資訊
try:
    print("-" * 50)
    print(f"🎉 轉換成功！")
    
    # 檢查主要量化模型
    if os.path.exists(TFLITE_MODEL):
        print(f"量化模型已儲存至 '{TFLITE_MODEL}'，檔案大小: {os.path.getsize(TFLITE_MODEL) / (1024 * 1024):.2f} MB")
    
    # 檢查浮點模型
    if os.path.exists("pointpillars_float.tflite"):
        print(f"浮點模型已儲存至 'pointpillars_float.tflite'，檔案大小: {os.path.getsize('pointpillars_float.tflite') / (1024 * 1024):.2f} MB")
    
    # 檢查基本模型
    if os.path.exists("pointpillars_basic.tflite"):
        print(f"基本模型已儲存至 'pointpillars_basic.tflite'，檔案大小: {os.path.getsize('pointpillars_basic.tflite') / (1024 * 1024):.2f} MB")
    
    # 檢查最小化模型
    if os.path.exists("pointpillars_minimal.tflite"):
        print(f"最小化模型已儲存至 'pointpillars_minimal.tflite'，檔案大小: {os.path.getsize('pointpillars_minimal.tflite') / (1024 * 1024):.2f} MB")
        
    print("\n注意：這些模型使用了 TF Select 操作，需要使用支援 TF Select 的 TFLite 運行時。")
    print("使用方法：在創建 Interpreter 時確保 TF Select 已啟用。")
    
except Exception as e:
    print(f"計算檔案大小時出錯: {e}")


# 提供一個簡單的測試函數來驗證TFLite模型
def test_tflite_model(tflite_model_path):
    print(f"\n=== 測試 TFLite 模型: {tflite_model_path} ===")
    try:
        # 加載 TFLite 模型
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # 獲取輸入輸出細節
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("模型輸入:")
        for input_detail in input_details:
            print(f"  - 名稱: {input_detail['name']}, 形狀: {input_detail['shape']}, 類型: {input_detail['dtype']}")
        
        print("模型輸出:")
        for output_detail in output_details:
            print(f"  - 名稱: {output_detail['name']}, 形狀: {output_detail['shape']}, 類型: {output_detail['dtype']}")
        
        print("TFLite 模型驗證成功！")
        return True
    except Exception as e:
        print(f"TFLite 模型驗證失敗: {e}")
        return False


# 主函數，方便直接執行
if __name__ == "__main__":
    print("\n" + "="*50)
    print("PointPillars 模型轉換到 TFLite 完成")
    print("="*50)
    
    # 嘗試測試已生成的模型
    if os.path.exists(TFLITE_MODEL):
        test_tflite_model(TFLITE_MODEL)
    
    if os.path.exists("pointpillars_float.tflite"):
        test_tflite_model("pointpillars_float.tflite")
    
    if os.path.exists("pointpillars_basic.tflite"):
        test_tflite_model("pointpillars_basic.tflite")
    
    print("\n轉換過程結束。您可以使用上述TFLite模型進行推理。")
    print("使用方法: 將模型加載到TFLite解釋器中，然後輸入處理後的點雲數據進行預測。")
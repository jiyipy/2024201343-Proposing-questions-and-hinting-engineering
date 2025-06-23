from pathlib import Path

# 构造增强版 notebook：在原有 notebook 上增加 3 个 tools 并注册
notebook_name = "智能化数据助手_yolov11版_final.ipynb"
notebook_path = Path("/mnt/data") / notebook_name

# 生成新的 Notebook 结构
with open("/mnt/data/智能化数据助手.ipynb", "r", encoding="utf-8") as f:
    content = f.read()

# 添加工具函数（在 get_user_info 后追加）
insert_code = """
# Tool: 缺陷检测
def detect_defect(image_path: str):
    import torch, os, json
    from datetime import datetime
    model_path = r"D:\\\\Users\\\\Lenovo\\\\PycharmProjects\\\\提出问题与提示工程\\\\runs\\\\detect\\\\train3\\\\weights\\\\best.pt"
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    results = model(image_path)
    results.print()
    results.show()
    log_dir = "defect_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "detect_log.jsonl")
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path,
        "detections": results.pandas().xyxy[0].to_dict(orient="records")
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\\n")
    return f"✅ 缺陷检测完成，日志已保存至 {log_file}"

# Tool: 模型训练
def train_yolo_model(data_path: str, use_old_data: bool, epochs: int, batch: int, image_size: int):
    import os
    from datetime import datetime
    old_data_flag = "--exist-ok" if use_old_data else ""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"train_log_{now}.txt"
    log_dir = "train_logs"
    os.makedirs(log_dir, exist_ok=True)
    command = (
        f"python train.py --img {image_size} --batch {batch} --epochs {epochs} "
        f"--data {data_path} --weights yolov11.pt --name yolov11_continue_{now} "
        f"{old_data_flag} > {os.path.join(log_dir, log_name)}"
    )
    os.system(command)
    return f"✅ 模型训练启动，日志保存在 {log_dir}/{log_name}"

# Tool: 日志查询
def query_logs(log_type: str = "detect", keyword: str = "", export: bool = False):
    import os, json
    import pandas as pd
    log_file = {
        "detect": "defect_logs/detect_log.jsonl",
        "train": "train_logs/train_log.txt"
    }.get(log_type, "defect_logs/detect_log.jsonl")
    if log_type == "detect":
        with open(log_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        filtered = [r for r in records if keyword in json.dumps(r, ensure_ascii=False)]
        df = pd.DataFrame(filtered)
    else:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        if keyword:
            content = "\\n".join([line for line in content.splitlines() if keyword in line])
        df = pd.DataFrame({"内容": content.splitlines()})
    if export:
        export_path = f"exported_{log_type}_log.xlsx"
        df.to_excel(export_path, index=False)
        return f"✅ 查询结果已导出：{export_path}"
    return df.head(10).to_markdown()
"""

# 插入函数注册段落
register_code = """
tools_list.update({
    "detect_defect": detect_defect,
    "train_yolo_model": train_yolo_model,
    "query_logs": query_logs
})
tools.extend([
    {'type': 'function',
     'function': {
        'name': 'detect_defect',
        'description': '使用YOLOv11模型对钢材图像进行缺陷检测并记录日志',
        'parameters': {'type': 'object',
                       'properties': {
                           'image_path': {'type': 'string', 'description': '图像路径'}
                       },
                       'required': ['image_path']
                      }
     }},
    {'type': 'function',
     'function': {
        'name': 'train_yolo_model',
        'description': '继续训练YOLOv11模型，支持自定义参数',
        'parameters': {'type': 'object',
                       'properties': {
                           'data_path': {'type': 'string'},
                           'use_old_data': {'type': 'boolean'},
                           'epochs': {'type': 'integer'},
                           'batch': {'type': 'integer'},
                           'image_size': {'type': 'integer'}
                       },
                       'required': ['data_path', 'use_old_data', 'epochs', 'batch', 'image_size']
                      }
     }},
    {'type': 'function',
     'function': {
        'name': 'query_logs',
        'description': '查询缺陷检测或训练日志，并可导出',
        'parameters': {'type': 'object',
                       'properties': {
                           'log_type': {'type': 'string'},
                           'keyword': {'type': 'string'},
                           'export': {'type': 'boolean'}
                       },
                       'required': ['log_type', 'keyword', 'export']
                      }
     }}
])
"""

# 注入代码到原始 notebook
content = content.replace("def get_user_info", insert_code + "\ndef get_user_info")
content = content.replace("tools = [", register_code + "\ntools = [")

# 保存为新文件
notebook_path.write_text(content, encoding="utf-8")
notebook_path.name

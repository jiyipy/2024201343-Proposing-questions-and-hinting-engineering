import os
import csv
from tqdm import tqdm

# 类别映射表 (根据NEU-DET官方定义)
CLASS_MAP = {
    0: "Rolled-in_Scale",
    1: "Patches",
    2: "Crazing",
    3: "Pitted_Surface",
    4: "Inclusion",
    5: "Scratches"
}


def extract_all_labels(root_path, output_file="NEU-DET_labels.csv"):
    """提取包含多目标的完整标签信息"""
    sets = ["train", "valid"]
    records = []

    # 遍历训练集和验证集
    for set_name in sets:
        label_dir = os.path.join(root_path, "NEU-DET", set_name, "labels")
        image_dir = os.path.join(root_path, "NEU-DET", set_name, "images")

        # 遍历所有标签文件
        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        for label_file in tqdm(label_files, desc=f"Processing {set_name}"):
            # 图像基本信息
            image_name = label_file.replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, image_name)

            # 解析标签文件 (每个图像可能包含多个目标)
            with open(os.path.join(label_dir, label_file), "r") as f:
                for idx, line in enumerate(f.readlines()):
                    try:
                        # 解析单行标签格式：class_id center_x center_y width height
                        parts = line.strip().split()
                        if len(parts) < 5: continue

                        record = {
                            "image_id": label_file.replace(".txt", ""),
                            "image_path": image_path,
                            "set_type": set_name,
                            "object_id": idx,  # 同一图像中的目标序号
                            "class_id": int(parts[0]),
                            "class_name": CLASS_MAP.get(int(parts[0]), "Unknown"),
                            "center_x": float(parts[1]),
                            "center_y": float(parts[2]),
                            "width": float(parts[3]),
                            "height": float(parts[4]),
                            "xmin": max(0, float(parts[1]) - float(parts[3]) / 2),  # 计算边界框
                            "ymin": max(0, float(parts[2]) - float(parts[4]) / 2),
                            "xmax": min(1, float(parts[1]) + float(parts[3]) / 2),
                            "ymax": min(1, float(parts[2]) + float(parts[4]) / 2)
                        }
                        records.append(record)
                    except Exception as e:
                        print(f"Error parsing {label_file} line {idx}: {e}")

    # 写入CSV文件
    if records:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"成功导出 {len(records)} 条标签记录至 {output_file}")
    else:
        print("未找到有效标签数据")


# 使用示例（替换为实际路径）
extract_all_labels(root_path="..")
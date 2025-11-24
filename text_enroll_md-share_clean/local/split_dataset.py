import json
import random
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Split datalist into train and validation sets.")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the full datalist file (e.g., datalist.english.json)")
    parser.add_argument("--train_ratio", type=float, default=0.95, 
                        help="Ratio of training data (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # 1. 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    print(f"Reading data from: {args.input_file}")
    
    # 2. 读取所有行
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total entries: {total_lines}")

    # 3. 随机打乱 (使用种子保证每次切分结果一样)
    random.seed(args.seed)
    random.shuffle(lines)

    # 4. 计算切分点
    split_idx = int(total_lines * args.train_ratio)
    
    train_lines = lines[:split_idx]
    valid_lines = lines[split_idx:]

    # 5. 构造输出文件名 (在原文件名后加 .train 和 .valid)
    base_dir = os.path.dirname(args.input_file)
    base_name = os.path.basename(args.input_file)
    # 移除 .json 后缀以便插入标识，或者直接追加
    if base_name.endswith('.json'):
        name_part = base_name[:-5] # 去掉 .json
        ext = '.json'
    else:
        name_part = base_name
        ext = ''
    
    train_file = os.path.join(base_dir, f"{name_part}.train{ext}")
    valid_file = os.path.join(base_dir, f"{name_part}.valid{ext}")

    # 6. 写入文件
    print(f"Writing train set ({len(train_lines)} entries) to: {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    print(f"Writing valid set ({len(valid_lines)} entries) to: {valid_file}")
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)

    print("\nDone! Split completed.")

if __name__ == "__main__":
    main()
"""
运行切分脚本
python /work104/lishuailong/data_processing/something_py/split_dataset.py \
    --input_file "/work104/lishuailong/librispeech_cleaned_data/datalist/datalist.english.json" \
    --train_ratio 0.9 \
    --seed 42
"""

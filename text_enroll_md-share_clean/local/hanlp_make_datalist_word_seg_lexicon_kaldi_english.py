# import re
# import sys
# import json
# import os
# import argparse
# from tqdm import tqdm

# def load_lexicon(lexicon_path: str) -> dict:
#     """
#     加载 LibriSpeech 英文发音词典。
#     文件格式: WORD P1 P2 P3...
#     例如: HELLO HH AH L OW
#     """
#     print(f"Loading lexicon from: {lexicon_path}")
#     lexicon_dict = {}
#     with open(lexicon_path, encoding='utf-8') as f:
#         for line in f:
#             # 转换为大写并移除空白符，以匹配LibriSpeech词典
#             parts = line.strip().upper().split()
#             if len(parts) < 2:
#                 continue
#             key = parts[0]
#             # 将音素列表保存为 list，方便后续处理
#             value = parts[1:] 
#             if key not in lexicon_dict:
#                 lexicon_dict[key] = value
#     print(f"Loaded {len(lexicon_dict)} words from lexicon.")
#     return lexicon_dict


# def read_scp(scp_file: str) -> dict:
#     """
#     读取Kaldi风格的scp文件 (如 wav_cut.scp, text_cut)
#     将其解析为 {key: content} 的字典
#     """
#     print(f"Reading scp file: {scp_file}")
#     obj = {}
#     with open(scp_file, encoding='utf-8') as ff:
#         for line in ff.readlines():
#             line = line.strip()
#             # 移除多个空格
#             line = re.sub(r"\s+", " ", line)
#             parts = line.split(" ", 1) # 按第一个空格分割
#             if len(parts) < 2:
#                 continue
#             utt_id = parts[0]
#             content = parts[1]
#             obj[utt_id] = content
#     print(f"Loaded {len(obj)} entries from {scp_file}.")
#     return obj


# def load_dict(dict_file: str) -> dict:
#     """
#     从文件加载或初始化 phone2id 映射表。
#     """
#     my_dict = {}
#     # 检查文件是否存在且不为空
#     if not os.path.exists(dict_file) or os.path.getsize(dict_file) == 0:
#         print(f"Info: Phone dict file '{dict_file}' not found or empty. ")
#         print("Initializing new map, reserving IDs 0, 1, 2. ")
#         # 根据文档 ，音素ID从3开始。我们为0,1,2保留特殊标记。
#         my_dict = {
#             '<pad>': 0, # 占位符
#             '<unk>': 1, # 未知音素
#             '<eos>': 2  # 序列结束符 (或其他特殊标记)
#         }
#         return my_dict
        
#     with open(dict_file, 'r', encoding='utf-8') as f_dict:
#         for line in f_dict:
#             parts = line.strip().split()
#             if len(parts) == 2:
#                 key, value = parts
#                 my_dict[key] = int(value)
#     print(f"Loaded {len(my_dict)} entries from {dict_file}.")
#     return my_dict


# def split_and_tokenize(text_obj: dict, in_phone2id: dict, in_lexicon: dict) -> (dict, dict):
#     """
#     (核心修改函数)
#     遍历所有英文文本，将其按词分割，并使用词典转换为音素ID序列。
#     """
#     phone2id = in_phone2id.copy() # 复制一份，避免直接修改原字典
#     tokenized_objs = {}    # 存储最终的JSON对象
    
#     # 获取当前最大的音素ID。
#     # 如果phone2id为空(或只含0,1,2)，max_phn_id将为2。
#     # 下一个新音素将被分配ID 3，这完全符合文档要求 。
#     if phone2id:
#         max_phn_id = max(phone2id.values())
#     else:
#         max_phn_id = 2 # 兜底，确保从3开始
    
#     print(f"Starting tokenization. Initial max phone ID: {max_phn_id}")
    
#     num_oov = 0
#     # 遍历 `text_cut` 中的每一行
#     for key, content in tqdm(text_obj.items(), desc="Tokenizing text"):
        
#         # 英文处理：按空格分割单词
#         # .upper() 是为了匹配LibriSpeech词典，它通常是全大写的
#         words = content.strip().upper().split()
        
#         if not words:
#             continue
            
#         # 这是按词分组的音素ID列表，例如: [[id, id], [id, id, id], ...]
#         # 这正是文档 [cite: 18] 中 'phn_label' 字段要求的格式
#         one_sample_phn_seg = [] 
#         is_oov = False # 标记是否遇到词典中没有的词
        
#         # 遍历这句话中的每一个英文单词
#         for word in words:
#             if word in in_lexicon:
#                 # 1. 单词在词典中
#                 # phones = ['HH', 'AH', 'L', 'OW']
#                 phones = in_lexicon[word]
#                 word_phn_ids = [] # 存储这个词的音素ID
                
#                 for phn in phones:
#                     # --- 动态ID分配 ---
#                     if phn not in phone2id:
#                         max_phn_id += 1
#                         phone2id[phn] = max_phn_id
#                         # 打印第一个被分配的音素，以供核对
#                         if max_phn_id == 3:
#                             print(f"Info: First new phone '{phn}' mapped to ID 3.")
#                     word_phn_ids.append(phone2id[phn])
                
#                 # 将这个词的音素ID列表作为一个整体，添加到句子列表中
#                 one_sample_phn_seg.append(word_phn_ids)
            
#             else:
#                 # 2. OOV (Out Of Vocabulary) 单词不在词典中
#                 if num_oov < 5: # 仅打印前5个OOV警告
#                     print(f"Warning: Skipping utterance '{key}'. Word '{word}' not in lexicon.")
#                 is_oov = True
#                 num_oov += 1
#                 break # 放弃处理这整句话
        
#         if is_oov:
#             continue
            
#         # 存储这句话的处理结果
#         tokenized_objs[key] = {
#             'segment_label': one_sample_phn_seg
#         }
        
#     print(f"Tokenization complete. Final max phone ID: {max_phn_id}")
#     if num_oov > 0:
#         print(f"Total utterances skipped due to OOV words: {num_oov}")
#     return tokenized_objs, phone2id


# def write_data_list(wav_scp: dict, tokenized_objs: dict, phone2id: dict, 
#                     out_phone2id: str, out_datalist: str):
#     """
#     将处理好的数据写入Datalist文件 (JSONL格式)。
#     同时保存更新后的 phone2id 映射表。
#     """
#     print(f"Writing datalist to {out_datalist}...")
    
#     with open(out_datalist, 'w', encoding='utf-8') as dlist:
#         for key, tokenized_item in tqdm(tokenized_objs.items(), desc="Writing datalist"):
#             if key not in wav_scp:
#                 # print(f"Warning: Skipping {key}, key not found in wav_scp.")
#                 continue
            
#             # 这就是我们需要的二维列表, e.g., [[3, 5, 4], [28, 1], ...] 
#             grouped_phn_label = tokenized_item['segment_label']
            
#             # 这就是文档要求的“单词占位符列表”，从0递增1 [cite: 19]
#             # 长度等于单词的数量
#             kw_candidate = list(range(len(grouped_phn_label)))
            
#             one_obj = {
#                 'key': key,
#                 'sph': wav_scp[key],
                
#                 # 遵循文档 [cite: 18] 的示例，'label' 和 'phn_label' 相同
#                 'label': grouped_phn_label,
#                 'phn_label': grouped_phn_label,
                
#                 'kw_candidate': kw_candidate
#             }
#             dlist.write(f"{json.dumps(one_obj, ensure_ascii=False)}\n")
    
#     # 写入更新后的 "音素" -> ID 映射文件
#     print(f"Writing new phone map to {out_phone2id}...")
#     with open(out_phone2id, 'w', encoding='utf-8') as pf:
#         # 按ID排序，方便查看
#         sorted_phones = sorted(phone2id.items(), key=lambda item: item[1])
#         for phn, id in sorted_phones:
#             pf.write(f"{phn} {id}\n")

# def main():
#     """
#     主函数：使用 argparse 解析命令行参数并编排处理流程。
#     """
#     parser = argparse.ArgumentParser(
#         description="Create English datalist for text_enroll_md project from LibriSpeech."
#     )
#     # --- 必需的参数 ---
#     parser.add_argument('--text_scp', type=str, required=True,
#                         help="Path to the input text_cut file.")
#     parser.add_argument('--wav_scp', type=str, required=True,
#                         help="Path to the input wav_cut.scp file.")
#     parser.add_argument('--in_lexicon', type=str, required=True,
#                         help="Path to the librispeech-lexicon.txt file. [cite: 15]")
#     parser.add_argument('--out_datalist', type=str, required=True,
#                         help="Path for the output datalist.txt (JSONL format).")
#     parser.add_argument('--out_phone2id', type=str, required=True,
#                         help="Path to write the new/updated phone-to-ID map.")
    
#     # --- 可选参数 ---
#     parser.add_argument('--in_phone2id', type=str, default="",
#                         help="Path to an *existing* phone-to-ID map. "
#                              "If provided, we will load it and continue adding new phones from there. "
#                              "If empty, we start fresh (0,1,2 reserved).")

#     args = parser.parse_args()

#     # 1. 读取 text_cut
#     text_scp = read_scp(args.text_scp)
    
#     # 2. 加载或初始化 phone2id 映射表
#     if args.in_phone2id and os.path.exists(args.in_phone2id):
#         print(f"Loading existing phone map from {args.in_phone2id}")
#         in_phone2id = load_dict(args.in_phone2id)
#     else:
#         # 如果没提供输入映射，就初始化一个新的
#         in_phone2id = load_dict("") 
    
#     # 3. 加载英文发音词典 
#     in_lexicon = load_lexicon(args.in_lexicon)
    
#     print(f"Total utterances to process: {len(text_scp)}")
    
#     # 4. !!! 执行核心处理 !!!
#     tokenized_objs, phone2id = split_and_tokenize(text_scp, in_phone2id, in_lexicon)
    
#     # 5. 读取 wav_cut.scp
#     wav_scp = read_scp(args.wav_scp)
    
#     # 6. 写入最终的 datalist 和 phone2id 映射表
#     write_data_list(wav_scp, tokenized_objs, phone2id, args.out_phone2id, args.out_datalist)
    
#     print("\nDatalist creation complete.")
#     print(f"  Output Datalist: {args.out_datalist}")
#     print(f"  Output Phone Map: {args.out_phone2id}")

# if __name__ == '__main__':
#     main()
import sys
import json
import os
import argparse
import re
from tqdm import tqdm

def load_lexicon(lexicon_path):
    """
    加载 LibriSpeech 英文发音词典。
    """
    print(f"Loading lexicon from: {lexicon_path}")
    lexicon = {}
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].upper()
            phones = parts[1:]
            lexicon[word] = phones
    print(f"Loaded {len(lexicon)} entries from lexicon.")
    return lexicon

def read_scp(scp_file):
    """读取 scp 文件"""
    print(f"Reading scp file: {scp_file}")
    data = {}
    with open(scp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                key, content = line.split(maxsplit=1)
                data[key] = content
            except ValueError:
                print(f"Warning: Skipping malformed line: {line}")
    print(f"Loaded {len(data)} entries.")
    return data

def load_or_init_phone_map(map_file):
    """加载或初始化 phone2id (ID 从 3 开始)"""
    phone2id = {}
    if os.path.exists(map_file) and os.path.getsize(map_file) > 0:
        print(f"Loading existing phone map: {map_file}")
        with open(map_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    phone2id[parts[0]] = int(parts[1])
    return phone2id

def build_and_save_aux_lexicon(lexicon, phone2id, output_path):
    """
    [核心功能] 构建 aux_lexicon.json
    包含：phone2id, by_len (按长度分组), confusion_matrix (预留)
    """
    print("Building aux_lexicon (Knowledge Base)...")
    
    aux_data = {
        "phone2id": phone2id,
        "by_len": {},          # 存放 { "3": [[p1, p2, p3], ...], ... }
        "confusion_matrix": {}, # [预留] 未来放置混淆矩阵
        "phones_list": []       
    }

    sorted_phones = sorted(phone2id.items(), key=lambda x: x[1])
    aux_data["phones_list"] = [p[0] for p in sorted_phones]

    for word, phones in tqdm(lexicon.items(), desc="Building by_len index"):
        try:
            phn_ids = [phone2id[p] for p in phones if p in phone2id]
            if len(phn_ids) != len(phones):
                continue
                
            length = str(len(phn_ids))
            if length not in aux_data["by_len"]:
                aux_data["by_len"][length] = []
            
            # 只存不重复的音素序列
            if phn_ids not in aux_data["by_len"][length]:
                aux_data["by_len"][length].append(phn_ids)
        except Exception:
            continue

    print(f"Writing aux_lexicon to {output_path} ...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aux_data, f, indent=4)
    print("Aux lexicon built successfully.")

def process_data(text_dict, wav_dict, lexicon, phone2id):
    # 确定当前最大 ID
    if phone2id:
        max_phn_id = max(phone2id.values())
    else:
        max_phn_id = 2 # 0,1,2 reserved
        
    output_list = []
    missing_words = set()
    
    print("Processing utterances...")
    for key, text in tqdm(text_dict.items()):
        if key not in wav_dict:
            continue
            
        wav_path = wav_dict[key]
        words = text.strip().upper().split()
        
        phn_label_groups = [] 
        valid_utterance = True
        
        for word in words:
            if word in lexicon:
                phones = lexicon[word]
                word_phone_ids = []
                for p in phones:
                    if p not in phone2id:
                        max_phn_id += 1
                        phone2id[p] = max_phn_id
                    word_phone_ids.append(phone2id[p])
                phn_label_groups.append(word_phone_ids)
            else:
                missing_words.add(word)
                valid_utterance = False
                break
        
        if valid_utterance and phn_label_groups:
            kw_candidate = list(range(len(phn_label_groups)))
            entry = {
                "key": key,
                "sph": wav_path,
                "label": phn_label_groups,
                "phn_label": phn_label_groups,
                "kw_candidate": kw_candidate
            }
            output_list.append(entry)
            
    if missing_words:
        print(f"Warning: {len(missing_words)} unique words skipped.")

    return output_list, phone2id

def main():
    parser = argparse.ArgumentParser(description="Generate full English datalist")
    parser.add_argument('--text_scp', required=True)
    parser.add_argument('--wav_scp', required=True)
    parser.add_argument('--in_lexicon', required=True)
    parser.add_argument('--out_datalist', required=True)
    parser.add_argument('--out_phone2id', required=True)
    parser.add_argument('--out_aux_lexicon', required=True)
    parser.add_argument('--in_phone2id', default="")
    args = parser.parse_args()

    text_dict = read_scp(args.text_scp)
    wav_dict = read_scp(args.wav_scp)
    lexicon = load_lexicon(args.in_lexicon)
    phone2id = load_or_init_phone_map(args.in_phone2id)
    
    datalist, final_phone2id = process_data(text_dict, wav_dict, lexicon, phone2id)
    
    print(f"Writing {len(datalist)} entries to {args.out_datalist}")
    with open(args.out_datalist, 'w', encoding='utf-8') as f:
        for entry in datalist:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Writing phone map to {args.out_phone2id}")
    sorted_phones = sorted(final_phone2id.items(), key=lambda item: item[1])
    with open(args.out_phone2id, 'w', encoding='utf-8') as f:
        for p, pid in sorted_phones:
            f.write(f"{p} {pid}\n")
            
    build_and_save_aux_lexicon(lexicon, final_phone2id, args.out_aux_lexicon)
    print("\nAll done!")

if __name__ == "__main__":
    main()

"""
python /work104/lishuailong/data_processing/something_py/hanlp_make_datalist_word_seg_lexicon_kaldi_english.py \
    --text_scp "/work104/lishuailong/librispeech_cleaned_data/cut_audio_output/text_cut" \
    --wav_scp "/work104/lishuailong/librispeech_cleaned_data/cut_audio_output/wav_cut.scp" \
    --in_lexicon "/work104/lishuailong/librispeech_cleaned_data/librispeech-lexicon.txt" \
    --out_datalist "/work104/lishuailong/librispeech_cleaned_data/datalist/datalist.english.json" \
    --out_phone2id "/work104/lishuailong/librispeech_cleaned_data/datalist/phone2id.english.txt" \
    --out_aux_lexicon "/work104/lishuailong/librispeech_cleaned_data/datalist/aux_lexicon_full.json"
"""
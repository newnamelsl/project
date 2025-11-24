import os
import sys
import argparse
import random
import textgrid
import librosa
import soundfile as sf
from tqdm import tqdm

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Cut audio files based on MFA TextGrid word timestamps."
    )
    parser.add_argument(
        "--textgrid_dir",
        type=str,
        required=True,
        help="MFA output directory containing .TextGrid files.",
    )
    parser.add_argument(
        "--wav_scp",
        type=str,
        required=True,
        help="Path to the original wav.scp file.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Path to the original text file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save cut audio files, wav_cut.scp, and text_cut.",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=2,
        help="Minimum number of words per segment.",
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=6,
        help="Maximum number of words per segment.",
    )
    parser.add_argument(
        "--extend_ms",
        type=int,
        default=0,
        help="Extend segment by this many milliseconds at start and end. (Default: 0)"
    )
    # [New] Random seed for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. (Default: 42)"
    )
    return parser.parse_args()


def load_scp_to_dict(path: str):
    """
    辅助函数：读取 scp 格式的文件并转为字典
    格式: key value
    """
    print(f"Loading scp: {path}")
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    # 只分割第一个空格，前面的做 ID，后面所有的做内容
                    key, content = line.split(maxsplit=1)
                    data[key] = content
                except ValueError:
                    print(f"Skipping malformed line in {path}: {line}")
    print(f"Loaded {len(data)} entries.")
    return data


def segment_audio(
    utt_id: str,
    audio_path: str,
    tg_path: str,
    text_line: str,
    output_wav_dir: str,
    min_words: int,
    max_words: int,
    extend_ms: int
):
    """
    核心逻辑：根据 TextGrid 将单个长音频切割为多个短片段
    """
    try:
        # 1. 读取 TextGrid 文件
        tg = textgrid.TextGrid.fromFile(tg_path)
    except Exception as e:
        print(f"Warning: Could not read TextGrid {tg_path}: {e}", file=sys.stderr)
        return []
    
    # 2. 获取 'words' 层
    try:
        word_tier = tg.getFirst("words")
    except ValueError:
        print(f"Warning: 'words' tier not found in {tg_path}. Skipping {utt_id}", file=sys.stderr)
        return []

    # 3. 过滤掉静音标记 (sil, sp, spn) 和空标记
    words = [
        interval
        for interval in word_tier
        if interval.mark and interval.mark.strip() not in ["", "sil", "sp", "spn"]
    ]

    if not words:
        # print(f"Warning: No words found in {tg_path} for {utt_id}", file=sys.stderr)
        return []
        
    # 4. 加载原始音频 (librosa 自动处理软连接)
    try:
        audio, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error: Could not load audio {audio_path}: {e}", file=sys.stderr)
        return []
    
    audio_duration_samples = len(audio)
    extend_samples = int((extend_ms / 1000.0) * sr)
    
    segments_info = []
    i = 0
    
    while i < len(words):
        # [关键点] 随机决定当前片段包含几个单词
        # 因为设置了随机种子，这里的"随机"是固定的
        num_words = random.randint(min_words, max_words)
        end_idx = min(i + num_words, len(words))
        
        segment_words = words[i:end_idx]
        
        # 5. 获取时间戳
        start_time = segment_words[0].minTime
        end_time = segment_words[-1].maxTime
        
        # 6. 转换时间戳为样本索引
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # 7. 应用延伸时间
        extended_start_sample = max(0, start_sample - extend_samples)
        extended_end_sample = min(audio_duration_samples, end_sample + extend_samples)
        
        # 8. 提取音频片段
        try:
            audio_segment = audio[extended_start_sample:extended_end_sample]
        except Exception as e:
            print(f"Error cutting audio {utt_id}: {e}", file=sys.stderr)
            i = end_idx
            continue
            
        # 9. 准备输出信息
        segment_id = f"{utt_id}_cut{i:04d}"
        segment_text = " ".join([word.mark.upper() for word in segment_words])
        output_wav_path = os.path.join(output_wav_dir, f"{segment_id}.wav")
        
        # 10. 保存音频片段
        try:
            sf.write(output_wav_path, audio_segment, sr)
            segments_info.append((segment_id, segment_text, output_wav_path))
        except Exception as e:
            print(f"Error writing wav {output_wav_path}: {e}", file=sys.stderr)
        
        i = end_idx
        
    return segments_info


def main():
    args = get_args()
    
    # [新增] 设置随机种子，确保结果可复现
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # 1. 准备输出目录
    output_wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(output_wav_dir, exist_ok=True)
    
    wav_cut_scp_path = os.path.join(args.output_dir, "wav_cut.scp")
    text_cut_path = os.path.join(args.output_dir, "text_cut")
    
    # 2. 加载输入文件
    wav_scp_dict = load_scp_to_dict(args.wav_scp)
    text_dict = load_scp_to_dict(args.text)
     
    print(f"Found {len(wav_scp_dict)} utterances to process.")
    
    # 3. 打开输出文件
    with open(wav_cut_scp_path, "w", encoding="utf-8") as f_wav_scp, \
         open(text_cut_path, "w", encoding="utf-8") as f_text_cut:
        
        # 4. 迭代处理
        for utt_id in tqdm(wav_scp_dict.keys(), desc="Cutting Audio"):
            if utt_id not in text_dict:
                continue
                
            audio_path = wav_scp_dict[utt_id]
            text_line = text_dict[utt_id]
            
            # -----------------------------------------------------------
            # [简化路径查找] Simplified path logic for Flat structure
            # -----------------------------------------------------------
            tg_path = os.path.join(args.textgrid_dir, f"{utt_id}.TextGrid")
            
            if not os.path.exists(tg_path):
                # print(f"Warning: TextGrid not found: {tg_path}", file=sys.stderr)
                continue

            # 切割音频
            segments = segment_audio(
                utt_id,
                audio_path,
                tg_path,
                text_line,
                output_wav_dir,
                args.min_words,
                args.max_words,
                args.extend_ms
            )
            
            # 5. 写入 scp 和 text
            for segment_id, segment_text, segment_path in segments:
                f_wav_scp.write(f"{segment_id} {segment_path}\n")
                f_text_cut.write(f"{segment_id} {segment_text}\n")
    
    print("\nAudio cutting complete.")
    print(f"Output wavs      : {output_wav_dir}")
    print(f"Output wav.scp   : {wav_cut_scp_path}")
    print(f"Output text      : {text_cut_path}")


if __name__ == "__main__":
    main()

"""
/work104/lishuailong/data_processing/something_py/cut_audio_text_by_timestamp_english.py \
    --textgrid_dir "/work104/lishuailong/librispeech_cleaned_data/mfa_output_ALL" \
    --wav_scp "/work104/lishuailong/librispeech_cleaned_data/wav.scp" \
    --text "/work104/lishuailong/librispeech_cleaned_data/text" \
    --output_dir "/work104/lishuailong/librispeech_cleaned_data/cut_audio_output" \
    --min_words 2 \
    --max_words 6 \
    --extend_ms 100 \
    --seed 42


"""
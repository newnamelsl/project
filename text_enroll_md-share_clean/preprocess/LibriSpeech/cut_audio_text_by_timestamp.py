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
    return parser.parse_args()


def load_scp_to_dict(path: str) -> Dict[str, str]:
    """将 scp 文件加载为字典 {key: path}"""
    print(f"Loading scp: {path}")
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    key, file_path = line.split(maxsplit=1)
                    data[key] = file_path
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
) -> List[Tuple[str, str]]:
    """
    根据 TextGrid 将单个音频切割为 2-6 词的片段
    返回 (segment_id, segment_text) 列表
    """
    try:
        # 1. 加载 TextGrid
        tg = textgrid.TextGrid.fromFile(tg_path)
    except Exception as e:
        print(f"Warning: Could not read TextGrid {tg_path}: {e}", file=sys.stderr)
        return []
    
    # 2. 找到 'words' tier 并提取所有非静音的单词
    try:
        word_tier = tg.getFirst("words")
    except ValueError:
        print(f"Warning: 'words' tier not found in {tg_path}. Skipping {utt_id}", file=sys.stderr)
        return []

    words = [
        interval
        for interval in word_tier
        if interval.mark and interval.mark.strip() not in ["", "sil", "sp", "spn"]
    ]

    if not words:
        print(f"Warning: No words found in {tg_path} for {utt_id}", file=sys.stderr)
        return []
        
    # 3. 加载原始音频
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
        # 4. 随机决定片段长度
        num_words = random.randint(min_words, max_words)
        end_idx = min(i + num_words, len(words))
        
        segment_words = words[i:end_idx]
        
        # 5. 计算时间戳
        start_time = segment_words[0].minTime
        end_time = segment_words[-1].maxTime
        
        # 6. 转换时间戳为样本索引
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # 7. 应用延伸时间 (extend_ms)
        extended_start_sample = max(0, start_sample - extend_samples)
        extended_end_sample = min(audio_duration_samples, end_sample + extend_samples)
        
        # 8. 提取音频片段
        try:
            audio_segment = audio[extended_start_sample:extended_end_sample]
        except Exception as e:
            print(f"Error cutting audio {utt_id}: {e}", file=sys.stderr)
            i = end_idx
            continue
            
        # 9. 准备输出
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
    
    # 1. 准备输出目录
    output_wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(output_wav_dir, exist_ok=True)
    
    wav_cut_scp_path = os.path.join(args.output_dir, "wav_cut.scp")
    text_cut_path = os.path.join(args.output_dir, "text_cut")
    
    # 2. 加载输入文件
    wav_scp_dict = load_scp_to_dict(args.wav_scp)
    text_dict = load_scp_to_dict(args.text) # kaldi 'text' file format is 'key text'
    
    print(f"Found {len(wav_scp_dict)} utterances to process.")
    
    # 3. 打开输出文件
    with open(wav_cut_scp_path, "w", encoding="utf-8") as f_wav_scp, \
         open(text_cut_path, "w", encoding="utf-8") as f_text_cut:
        
        # 4. 迭代处理每个音频
        for utt_id in tqdm(wav_scp_dict.keys(), desc="Cutting Audio"):
            if utt_id not in text_dict:
                print(f"Warning: Skipping {utt_id}, missing in text file.", file=sys.stderr)
                continue
                
            audio_path = wav_scp_dict[utt_id]
            text_line = text_dict[utt_id]
            
            # 构建 TextGrid 文件路径
            # 假设 TextGrid 的相对路径与 wav 文件一致
            # 例如: .../39/121914/39-121914-0013.wav -> .../39/121914/39-121914-0013.TextGrid
            # MFA 通常会在其输出目录中保持原始的目录结构
            
            # 这是一个常见的MFA输出结构，如果您的MFA输出了扁平结构，您可能需要调整
            # 假设 train_all_960/wav.scp 中的 key 是 '39-121914-0013'
            # 并且 MFA 输出在 mfa_output_librispeech/39/121914/39-121914-0013.TextGrid
            # 
            # 让我们简化：假设 MFA 输出目录中就是以 utt_id 命名的 .TextGrid 文件
            # 比如 /work104/lishuailong/mfa_output_librispeech/39-121914-0013.TextGrid
            # 
            # 您的原始 wav.scp 里的 key 可能是 '39-121914-0013'
            # 您的 TextGrid 目录里 *可能* 也是 '39-121914-0013.TextGrid'
            # 
            # 更正：MFA 输出目录会包含一个与语料库同名的子目录
            # 例如: mfa align corpus_dir ... output_dir
            # 输出会是 output_dir/corpus_dir_aligned/....
            # 
            # 让我们假设 `args.textgrid_dir` 是包含 .TextGrid 文件的 *根目录*
            # 比如 /work104/lishuailong/mfa_output_librispeech/
            # MFA 会在里面创建与您输入目录匹配的子文件夹
            
            # LibriSpeech 的 key '39-121914-0013' 格式, 对应的文件路径是 '.../39/121914/39-121914-0013.flac'
            # MFA 的 TextGrid 输出路径通常会是 `textgrid_dir/39/121914/39-121914-0013.TextGrid`
            parts = utt_id.split("-")
            if len(parts) == 3:
                tg_sub_path = os.path.join(parts[0], parts[1], f"{utt_id}.TextGrid")
                tg_path = os.path.join(args.textgrid_dir, tg_sub_path)
            else:
                # 备用方案：如果 key 不是标准 LibriSpeech 格式，就在根目录找
                tg_path = os.path.join(args.textgrid_dir, f"{utt_id}.TextGrid")
            
            if not os.path.exists(tg_path):
                # 备用方案2：MFA V2.x 可能会把所有文件平铺在输出目录
                tg_path_flat = os.path.join(args.textgrid_dir, f"{utt_id}.TextGrid")
                if os.path.exists(tg_path_flat):
                    tg_path = tg_path_flat
                else:
                    # 备用方案3：MFA V3.x
                    tg_path_3x = os.path.join(args.textgrid_dir, "alignments", tg_sub_path)
                    if os.path.exists(tg_path_3x):
                         tg_path = tg_path_3x
                    else:
                        print(f"Warning: Cannot find TextGrid for {utt_id}. Tried: \n - {tg_path}\n - {tg_path_flat}\n - {tg_path_3x}", file=sys.stderr)
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
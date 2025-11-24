import os
import yaml
import json
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import sys

from yamlinclude import YamlIncludeConstructor
from local.utils import make_dict_from_file
from model import m_dict
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pypinyin import pinyin, Style
import numpy as np
import copy
import time

test_config_file = sys.argv[1]
# test_sph_batch = int(sys.argv[2])
# test_cross_batch = int(sys.argv[3])
# custom_keyword_num = int(sys.argv[4])
rtf_output = sys.argv[2]

custom_keyword_nums = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 96]
#custom_keyword_nums = [1, 2, 4, 8, 16, 32, 64]
total_batch = 96
#total_batch = 64
speech_batch_size = total_batch


shift_per_hour = 10 * 3600
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
test_config = yaml.load(open(test_config_file), Loader=yaml.FullLoader)
testset_dir = test_config['testset_dir']
train_config = test_config.get('train_config', None)
train_datalist = train_config['data_config']['data_list']
train_data_dir = os.path.dirname(train_datalist)
test_result_dir = test_config['test_result_dir']
exp_dir = train_config['exp_config']['exp_dir']
trained_ckpt_path = os.path.join(exp_dir, 'kwatt_asr_avg.pt')
if 'trained_ckpt' in test_config:
    trained_ckpt_path = test_config['trained_ckpt']

word2id = make_dict_from_file(train_data_dir+'/word2id.txt')
id2word = {int(v): k for k, v in word2id.items()}
chunk_shift = test_config['chunk_shift']
chunk_size = test_config['chunk_size']
plot_each_word = test_config.get('plot_each_word', False)
test_option = test_config['test_option']
pos_shift = test_option['pos_shift']
neg_shift = test_option['neg_shift']
use_pos_pos_trials = test_option['use_pos_pos_trials']
count_top1_keyword = test_option['count_top1_keyword']
count_best_chunk = test_option['count_best_chunk']

dcf_config = test_config.get('dcf_config', None)
if dcf_config is not None:
    cost_miss = dcf_config['cost_miss']
    cost_fa = dcf_config['cost_fa']
    prior_target = dcf_config['prior_target']

# test_batch = 1
#test_batch = test_config.get('test_batch', 10)
# print("test_sph_batch: {}".format(test_sph_batch))
# print("test_cross_batch: {}".format(test_cross_batch))

os.makedirs(test_result_dir, exist_ok=True)

def backup_configs():
     # train config backup in result dir
     train_f = open("{}/train.yaml".format(test_result_dir), 'w')
     yaml.dump(train_config, train_f)
     # test config backup in result dir
     test_f = open("{}/test.yaml".format(test_result_dir), 'w')
     yaml.dump(test_config, test_f)

def make_fenyinta():
    positive_scp = make_dict_from_file(testset_dir+'/positive.wav.scp')
    negative_scp = make_dict_from_file(testset_dir+'/negative.wav.scp')
    positive2target = make_dict_from_file(testset_dir+'/positive.txt')
    keyword2phn = make_dict_from_file(testset_dir+'/keyword2phnseq')
    phn2id = make_dict_from_file(train_data_dir+'/phone2id.txt')

    keyword2phnid = {k: [phn2id[p] for p in v.split(" ")] for k, v in keyword2phn.items()}
    keyword2id = {k: i for i, k in enumerate(keyword2phnid.keys())}
    save_material = {
        'keyword2phn': keyword2phn,
        'keyword2phnid': keyword2phnid
    }
    meta_path = os.path.join(test_result_dir, 'fenyinta_meta.pt')
    torch.save(save_material, meta_path)
    positive_list_writer = open(os.path.join(test_result_dir, 'positive.datalist.txt'), 'w')
    for utt, wav in positive_scp.items():
        one_obj = dict(
            key=utt,
            sph=wav,
            target_keyword_id=keyword2id[positive2target[utt]],
            target_keyword = positive2target[utt]
        )
        positive_list_writer.write("{}\n".format(json.dumps(one_obj)))
    
    negative_list_writer = open(os.path.join(test_result_dir, 'negative.datalist.txt'), 'w')
    for utt, wav in negative_scp.items():
        one_obj = dict(
            key=utt,
            sph=wav
        )
        negative_list_writer.write("{}\n".format(json.dumps(one_obj)))


FBANK_DEFAULT_SETTING = {
    'num_mel_bins': 40, 'frame_length': 25, 'frame_shift': 10
}

def read_data_list(fname):
    dlist = []
    with open(fname) as tf:
        for line in tf.readlines():
            line = line.strip()
            mix_obj = json.loads(line)
            dlist.append(mix_obj)
    return dlist



def load_model():
    trained_ckpt = torch.load(trained_ckpt_path)
    trained_ckpt = trained_ckpt['model']

    model_arch = test_config['test_model_arch']
    #model_arch = train_config['model_arch']
    model_config = train_config['model_config']
    model = m_dict[model_arch]

    test_model = model(**model_config)
    test_model.load_state_dict(trained_ckpt)
    test_model.to('cuda:0')
    test_model.eval()
    return test_model

def make_batch_keyword(keyword2id):

    keyword_list = []
    keyword_len = []
    idx2keyword = {}
    keyword2idx = {}
    for i, (keyword, phn_seq) in enumerate(keyword2id.items()):
        idx2keyword[i] = keyword
        keyword2idx[keyword] = i
        phn_seq = [1] + phn_seq + [2]
        phn_seq = torch.tensor(phn_seq).to(torch.long)
        keyword_list.append(phn_seq)
        keyword_len.append(len(phn_seq))
    keyword_list = pad_sequence(keyword_list, batch_first=True, padding_value=0)
    # print(keyword_list)
    # print(torch.tensor(keyword_len))
    return (keyword_list, torch.tensor(keyword_len), idx2keyword, keyword2idx)


def print_feat(feat):
    with open("test_feat_print.txt", 'a') as f_feat:
        for i, keyword_feat in enumerate(feat):
            for j, frame_feat in enumerate(keyword_feat):
                f_feat.write("\n{}: {}: [".format(i, j))
                for r, f in enumerate(frame_feat):
                    if r == 0:
                        f_feat.write("{}".format(f))
                    else:
                        f_feat.write(", {}".format(f))
                f_feat.write("]")


def inference_shift_batch_time(model, test_list, keyword_material, idx2keyword, keyword2idx, shift=False, chunk_size=chunk_size, speech_batch_size=1, cross_batch_size=1, custom_keyword_num=1):
    print("speech batch size: {}, cross batch size: {}, custom keyword num: {}".format(speech_batch_size, cross_batch_size, custom_keyword_num))
    t_sum_sph_emb = 0
    t_sum_cross_att = 0
    t_sum_kw_emb = 0
    t_sum_total = 0
    t_start_total = time.time()
    keyword, keyword_len = keyword_material
    num_keyword = keyword.size(0)
    result = {}
    utt2keyword_idx = {}
    # print("input keyword: {}, 2nd keyword: {}".format(keyword, keyword[1]))
    selected_keyword = keyword[1]
    selected_keyword_len = keyword_len[1]
    keyword = torch.tensor([selected_keyword.tolist() for x in range(custom_keyword_num)]).to(torch.long)
    keyword_len = torch.tensor([selected_keyword_len for x in range(custom_keyword_num)]).to(torch.long)
    print("keyword: {}".format(keyword))
    num_keyword = custom_keyword_num
    input_keyword = (keyword, keyword_len)
    input_keyword = (d.to('cuda:0') for d in input_keyword)
    t_start_kw_emb = time.time()
    keyword_embedding, keyword_mask = model.evaluate_kw_emb(input_keyword)
    t_sum_kw_emb = time.time() - t_start_kw_emb
    # print("keyword embedding size: {}, keyword mask size: {}".format(keyword_embedding.size(), keyword_mask.size()))
    sum_duration = 0
    for i, obj in enumerate(test_list):
        if i < 50:
            print("utt: {}".format(i))
        elif i % 300 == 0:
            print("utt: {}".format(i))
        key = obj['key']
        if 'target_keyword' in obj:
            utt2keyword_idx[key] = keyword2idx[obj['target_keyword']]
        wav = torchaudio.load(obj['sph'])[0]
        wav_dur = wav.size(1) / 16000
        print("wav duration: {}".format(wav_dur))
        sum_duration += wav_dur
        fbank = kaldi.fbank(wav, **FBANK_DEFAULT_SETTING)
        fbank = fbank.unsqueeze(0)
        j = 0
        result[key] = []
        if not shift:
            chunk_size = fbank.size(1)
        input_batch = []
        # speech_batch_size = test_sph_batch
        # cross_batch_size = test_cross_batch
        speech_batch_len = 0
        cross_batch_len = 0
        assert speech_batch_size % cross_batch_size == 0 and speech_batch_size >= cross_batch_size
        chunk_len = torch.tensor([chunk_size]).to('cuda:0')
        while j * chunk_shift + chunk_size <= fbank.size(1):
            if  j >= 1000 and j % 1000 == 0:
                print("chunk: {}".format(j))
            chunk = fbank[:, j * chunk_shift:j * chunk_shift + chunk_size, :]
            #print_feat(chunk)
            chunk = chunk.to('cuda:0')
            speech_input = (chunk, chunk_len)
            # speech_input = (d.to('cuda:0') for d in speech_input)
            # speech_input = tuple(speech_input)
            chunk, chunk_len = speech_input
            if speech_batch_len == 0:
                chunk_batch = copy.deepcopy(chunk)
                chunk_len_batch = copy.deepcopy(chunk_len)
            else:
                chunk_batch = torch.cat((chunk_batch, chunk), dim=0)
                chunk_len_batch = torch.cat((chunk_len_batch, chunk_len), dim=0)
            speech_batch_len += 1
            if speech_batch_len >= speech_batch_size:
                speech_input_batch = (chunk_batch, chunk_len_batch)
                # print("chunk size: {}, chunk_len: {}".format(chunk.size(), chunk_len))
                t_start_sph_emb = time.time()
                # print("start sph_emb")
                # for i in range(100):
                speech_embedding_batch, speech_mask_batch = model.evaluate_sph_emb(speech_input_batch)
                    # time.sleep(0.1)
                t_sum_sph_emb += time.time() - t_start_sph_emb
                # torch.cuda.empty_cache()
                cross_batch_len = 0
                while cross_batch_len + cross_batch_size <= speech_batch_len:
                    speech_embedding_batch_sub = speech_embedding_batch[cross_batch_len:cross_batch_len+cross_batch_size, :, :]
                    speech_mask_batch_sub = speech_mask_batch[cross_batch_len:cross_batch_len+cross_batch_size, :, :]
                    # print("speech embedding batch size: {}, start index: {}, end index: {}".format(speech_embedding_batch.shape, cross_batch_len, cross_batch_len+cross_batch_size))
                    # print("keyword embeding dim: {}, keyword mask dim: {}".format(keyword_embedding.size(), keyword_mask.size()))
                    keyword_embedding_batch = keyword_embedding.repeat(cross_batch_size, 1, 1)
                    keyword_mask_batch = keyword_mask.repeat(cross_batch_size, 1, 1)
                    
                    # print("speech embedding size: {}, speech mask size: {}, cross_batch_len: {}, cross_batch_size: {}".format(speech_embedding_batch.size(), speech_mask_batch.size(), cross_batch_len, cross_batch_size))
                    speech_embedding_batch_sub = speech_embedding_batch_sub.repeat_interleave(num_keyword, dim=0)
                    speech_mask_batch_sub = speech_mask_batch_sub.repeat_interleave(num_keyword, dim=0)
                    # print("speech embedding size: {}, speech mask size: {}, keyword embedding size: {}, keyword mask size: {}, num_keyword: {}".format(speech_embedding_batch.size(), speech_mask_batch.size(), keyword_embedding_batch.size(), keyword_mask_batch.size(), num_keyword))
                    cross_input_batch = (speech_embedding_batch_sub, speech_mask_batch_sub, keyword_embedding_batch, keyword_mask_batch)
                    # det_result_batch, hyp_batch = model.evaluate(input_data_batch)
                    t_start_cross_att = time.time()
                    # print("start cross attention")
                    # for i in range(100):
                    det_result_batch, hyp_batch = model.evaluate_cross_attention(cross_input_batch)
                        # time.sleep(0.1)
                    t_sum_cross_att += time.time() - t_start_cross_att
                    # torch.cuda.empty_cache()

                    #max_idx = torch.argmax(det_result).item()
                    # 将 det_result_batch 分组
                    det_result_batches = det_result_batch.view(-1, num_keyword, det_result_batch.size(1)).to('cpu')
                    # 直接转换为列表并存储
                    result[key].extend(det_result_batches.tolist())
                    # det_result_batches = det_result_batch.view(-1, num_keyword, det_result_batch.size(1))
                    cross_batch_len += cross_batch_size
                    # speech_batch_len -= cross_batch_size
                speech_batch_len = 0
            j += 1
        if speech_batch_len > 0:
            speech_input_batch = (chunk_batch, chunk_len_batch)
            # print("chunk size: {}, chunk_len: {}".format(chunk.size(), chunk_len))
            t_start_sph_emb = time.time()
            speech_embedding_batch, speech_mask_batch = model.evaluate_sph_emb(speech_input_batch)
            t_sum_sph_emb += time.time() - t_start_sph_emb
            # torch.cuda.empty_cache()
            cross_batch_len = 0
            while cross_batch_len + cross_batch_size <= speech_batch_len:
                speech_embedding_batch_sub = speech_embedding_batch[cross_batch_len:cross_batch_len+cross_batch_size, :, :]
                speech_mask_batch_sub = speech_mask_batch[cross_batch_len:cross_batch_len+cross_batch_size, :, :]
                keyword_embedding_batch = keyword_embedding.repeat(cross_batch_size, 1, 1)
                keyword_mask_batch = keyword_mask.repeat(cross_batch_size, 1, 1)
                
                # print("speech embedding size: {}, speech mask size: {}".format(speech_embedding.size(), speech_mask.size()))
                speech_embedding_batch_sub = speech_embedding_batch_sub.repeat_interleave(num_keyword, dim=0)
                speech_mask_batch_sub = speech_mask_batch_sub.repeat_interleave(num_keyword, dim=0)
                cross_input_batch = (speech_embedding_batch_sub, speech_mask_batch_sub, keyword_embedding_batch, keyword_mask_batch)

                # det_result_batch, hyp_batch = model.evaluate(input_data_batch)
                t_start_cross_att = time.time()
                det_result_batch, hyp_batch = model.evaluate_cross_attention(cross_input_batch)
                t_sum_cross_att += time.time() - t_start_cross_att
                # torch.cuda.empty_cache()

                # 将 det_result_batch 分组
                det_result_batches = det_result_batch.view(-1, num_keyword, det_result_batch.size(1)).to('cpu')
                # 直接转换为列表并存储
                result[key].extend(det_result_batches.tolist())

                cross_batch_len += cross_batch_size

            if cross_batch_len < speech_batch_len:
                speech_embedding_batch_sub = speech_embedding_batch[cross_batch_len:, :, :]
                speech_mask_batch_sub = speech_mask_batch[cross_batch_len:, :, :]
                keyword_embedding_batch = keyword_embedding.repeat(speech_embedding_batch_sub.size(0), 1, 1)
                keyword_mask_batch = keyword_mask.repeat(speech_embedding_batch_sub.size(0), 1, 1)
                # print("speech embedding size: {}, speech mask size: {}".format(speech_embedding.size(), speech_mask.size()))
                speech_embedding_batch_sub = speech_embedding_batch_sub.repeat_interleave(num_keyword, dim=0)
                speech_mask_batch_sub = speech_mask_batch_sub.repeat_interleave(num_keyword, dim=0)
                # input_data_batch = (chunk_batch, chunk_len_batch, keyword_batch, keyword_len_batch)
                print("remaining batch: speech embedding batch sub size: {}, speech mask batch sub size: {}, keyword embedding batch size: {}, keyword mask batch size: {}".format(speech_embedding_batch_sub.size(), speech_mask_batch_sub.size(), keyword_embedding_batch.size(), keyword_mask_batch.size()))
                cross_input_batch = (speech_embedding_batch_sub, speech_mask_batch_sub, keyword_embedding_batch, keyword_mask_batch)

                # det_result_batch, hyp_batch = model.evaluate(input_data_batch)
                t_start_cross_att = time.time()
                det_result_batch, hyp_batch = model.evaluate_cross_attention(cross_input_batch)
                t_sum_cross_att += time.time() - t_start_cross_att
                # torch.cuda.empty_cache()

                # 将 det_result_batch 分组
                det_result_batches = det_result_batch.view(-1, num_keyword, det_result_batch.size(1)).to('cpu')
                # 直接转换为列表并存储
                result[key].extend(det_result_batches.tolist())


    t_sum_total = time.time() - t_start_total
    return result, utt2keyword_idx, t_sum_kw_emb, t_sum_sph_emb, t_sum_cross_att, t_sum_total, sum_duration


if __name__ == '__main__':
    
    backup_configs()
    result_id = '_'.join([str(int(pos_shift)), str(int(neg_shift))])
    analysis_id = '_'.join([str(int(pos_shift)), str(int(neg_shift)), str(int(use_pos_pos_trials)), str(int(count_top1_keyword)), str(int(count_best_chunk))])
    result_path = os.path.join(test_result_dir, 'test.result_{}.pt'.format(result_id))
    meta_path = os.path.join(test_result_dir, 'fenyinta_meta.pt')
    #if not os.path.exists(meta_path): 
        # 如果第一次做测试，需要将测试数据的一些信息存储在一个统一格式的文件fenyinta_meta.pt，方便后续测试
        # 直接使用
    make_fenyinta()
    test_material = torch.load(meta_path)
    model = load_model()
    keyword2id = {k: [int(vv) for vv in v] for k, v in test_material['keyword2phnid'].items()}
    keyword, keyword_len, idx2keyword, keyword2idx = make_batch_keyword(keyword2id)

    negative_list = read_data_list(os.path.join(test_result_dir, 'negative.datalist.txt'))
    negative_result_path = os.path.join(test_result_dir, 'test.result_neg_{}.pt'.format(int(neg_shift)))
    #if os.path.exists(negative_result_path):
    #    test_result = torch.load(negative_result_path)
    #    negative_result = test_result['neg_hyp']
    #else:
    t_start = time.time()
    # negative_result, _, t_cost_kw_emb, t_cost_sph_emb, t_cost_cross_att, t_cost_total = inference_shift_batch_time(model, negative_list, (keyword, keyword_len), idx2keyword, keyword2idx, shift=neg_shift, custom_keyword_num=custom_keyword_num)
    with open(rtf_output, 'w') as f:
        f.write("keyword_num, kw_emb, sph_emb, cross_att, total, duration, rtf\n")
        for custom_keyword_num in custom_keyword_nums:
            assert total_batch % custom_keyword_num == 0
            cross_batch_size = int(total_batch / custom_keyword_num)
            negative_result, _, t_cost_kw_emb, t_cost_sph_emb, t_cost_cross_att, t_cost_total, sum_duration = inference_shift_batch_time(model, negative_list, (keyword, keyword_len), idx2keyword, keyword2idx, shift=neg_shift, speech_batch_size=speech_batch_size, cross_batch_size=cross_batch_size, custom_keyword_num=custom_keyword_num)
            f.write("{}, {}, {}, {}, {}, {}, {}\n".format(custom_keyword_num, t_cost_kw_emb, t_cost_sph_emb, t_cost_cross_att, t_cost_total, sum_duration, (t_cost_sph_emb + t_cost_cross_att) / sum_duration))
    # print("time cost for negative reference: {:.3f}\n".format(time.time() - t_start))
    # print("time cost for kw_emb: {:.3f}\n".format(t_cost_kw_emb))
    # print("time cost for sph_emb: {:.3f}\n".format(t_cost_sph_emb))
    # print("time cost for cross_att: {:.3f}\n".format(t_cost_cross_att))
    # print("time cost for total: {:.3f}\n".format(t_cost_total))
    # print("time cost for ohters: {:.3f}\n".format(t_cost_total - t_cost_kw_emb - t_cost_sph_emb - t_cost_cross_att))
    # print("rtf: {:.3f}".format((t_cost_sph_emb + t_cost_cross_att) / sum_duration))
    # print("")
    # negative_test_result = {
    #     'neg_hyp': negative_result
    # }
    # negative_result_json_path = "{}.{}-{}.json".format(negative_result_path, test_sph_batch, test_cross_batch)
    # with open(negative_result_json_path, 'w') as f:
    #     for k, v in negative_result.items():
    #         for chunk in v:
    #             for score in chunk:
    #                 f.write("{}\n".format(score))
        # json.dump(negative_test_result, f)
    #torch.save(negative_test_result, negative_result_path)
    
    ## 由于不同的阈值会产生不同的 Recall 和 FA，所以我这里将每一条语言在每个关键词上的分数都存在了test.result.pt 中，
    ## 方便后续确定每个词的阈值。
    #test_result = {
    #    'pos_hyp': positive_result, 'neg_hyp': negative_result, 'pos_utt2keyword_idx': utt2keyword_idx,
    #    'idx2keyword': idx2keyword
    #}
    ##最终会根据测试结果统一画一个 roc 曲线。
    #out_csv = os.path.join(test_result_dir, "dcf_{}_{}_{}_{}.csv".format(cost_miss, cost_fa, prior_target, analysis_id))
    #f_out_csv = open(out_csv, 'w')
    #result_analysis_shift(positive_result, utt2keyword_idx, negative_result, test_result_dir, analysis_id, f_out_csv)
    #if plot_each_word:
    #    for keyword, keywordidx in keyword2idx.items():
    #        print(keyword, keywordidx)
    #        result_analysis_shift_one_word(positive_result, utt2keyword_idx, negative_result, test_result_dir, analysis_id, keyword, keywordidx, f_out_csv)
    #f_out_csv.close()

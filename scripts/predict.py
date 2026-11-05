"""
零数据库独立推理脚本 (Predict Script)

无需任何 RocksDB / count 数据库，无需预计算的 scores 文件！
给定输入的 FASTA 文件，直接输出全 7 级层次化分类预测结果与置信度概率。

用法:
  python scripts/predict.py \
      --config configs/finetune_bacteria.yaml \
      --checkpoint runs/bacteria_species_cls_v1_finetune/best_model.pt \
      --fasta test_sample.fasta \
      --output predictions.tsv
"""

import sys
import os
import argparse
import pickle
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model.model import build_model
from src.data.tokenizer import KmerTokenizer
from src.data.taxonomy import TaxonomyDB, TAXONOMY_LEVELS


def read_fasta_simple(fasta_path: str):
    """简单解析 FASTA 文件"""
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)

        if current_header is not None:
            sequences.append((current_header, "".join(current_seq)))

    return sequences


@torch.no_grad()
def predict_sequence(model, sequence: str, tokenizer: KmerTokenizer,
                     target_token_len: int, taxonomy_db: TaxonomyDB,
                     device: torch.device):
    """
    对单条 DNA 序列进行预测 (零数据库模式)
    """
    # 1. 21-mer 分词 (padding 到 4076)
    bases_21, pad_mask = tokenizer.tokenize_with_padding(sequence, target_token_len)
    
    # 2. 转换为 Tensor: batch_size = 1
    bases_21_tensor = torch.from_numpy(bases_21).long().unsqueeze(0).to(device)  # [1, L, 21]
    
    # 3. 前向传播 (uniqueness_scores=None -> 自动补全 0.0)
    logits, Z, _, _ = model.forward_finetune(
        bases_21=bases_21_tensor,
        uniqueness_scores=None  # 零数据库极速推理模式！
    )

    # 4. 解析预测概率与标签
    predictions = {}
    for level in TAXONOMY_LEVELS:
        if level in logits:
            level_logits = logits[level][0]  # [C_level]
            probs = F.softmax(level_logits, dim=-1)
            pred_idx = torch.argmax(probs).item()
            prob_val = probs[pred_idx].item()

            # 解码为物种/分类名称
            pred_label = taxonomy_db.idx2label[level][pred_idx] if (
                level in taxonomy_db.idx2label and pred_idx < len(taxonomy_db.idx2label[level])
            ) else f"Unknown_idx_{pred_idx}"

            predictions[level] = {
                'label': pred_label,
                'confidence': float(prob_val),
                'idx': pred_idx,
            }

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Zero-DB Fast Taxonomy Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--fasta', type=str, required=True, help='Input FASTA file to classify')
    parser.add_argument('--taxonomy-db', type=str, default=None, help='Path to taxonomy_db.pkl')
    parser.add_argument('--output', type=str, default=None, help='Output TSV file path')
    parser.add_argument('--gpus', type=str, default='0', help='GPU ID')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = torch.device(f'cuda:{args.gpus}' if torch.cuda.is_available() and args.gpus != '-1' else 'cpu')

    # 加载 TaxonomyDB
    taxonomy_path = args.taxonomy_db or os.path.join(config.data.index_dir, 'taxonomy_db.pkl')
    if not os.path.exists(taxonomy_path):
        raise FileNotFoundError(f"TaxonomyDB file not found at: {taxonomy_path}")
    taxonomy_db = TaxonomyDB.load(taxonomy_path)

    # 构建并加载模型
    print(f"[Predict] Building model in zero-database inference mode...")
    model = build_model(
        config=config.model,
        mode='finetune',
        num_classes_per_level=taxonomy_db.num_classes,
    )
    
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    tokenizer = KmerTokenizer(kmer_size=config.model.kmer_size)
    target_token_len = config.model.max_seq_len

    # 读取输入 FASTA
    records = read_fasta_simple(args.fasta)
    print(f"[Predict] Read {len(records)} sequences from {args.fasta}")

    results = []
    print("\n" + "=" * 80)
    print(f"{'Sequence Header':<30} | {'Predicted Species':<30} | {'Confidence':<10}")
    print("=" * 80)

    for header, seq in tqdm(records, desc="Classifying"):
        if len(seq) < config.model.kmer_size:
            print(f"Skipping sequence '{header}': length ({len(seq)}) < kmer_size ({config.model.kmer_size})")
            continue

        preds = predict_sequence(
            model=model,
            sequence=seq,
            tokenizer=tokenizer,
            target_token_len=target_token_len,
            taxonomy_db=taxonomy_db,
            device=device,
        )

        sp_info = preds.get('species', {'label': 'Unknown', 'confidence': 0.0})
        print(f"{header[:30]:<30} | {sp_info['label'][:30]:<30} | {sp_info['confidence']*100:6.2f}%")

        results.append({
            'header': header,
            'sequence_length': len(seq),
            'predictions': preds,
        })

    # 输出 TSV 结果
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            # 写入表头
            header_cols = ['Sequence_Header', 'Seq_Length'] + [f"{lvl}_pred" for lvl in TAXONOMY_LEVELS] + [f"{lvl}_conf" for lvl in TAXONOMY_LEVELS]
            f.write("\t".join(header_cols) + "\n")
            
            for res in results:
                row = [res['header'], str(res['sequence_length'])]
                # 预测类别
                for lvl in TAXONOMY_LEVELS:
                    row.append(res['predictions'].get(lvl, {}).get('label', 'N/A'))
                # 置信度
                for lvl in TAXONOMY_LEVELS:
                    row.append(f"{res['predictions'].get(lvl, {}).get('confidence', 0.0):.4f}")
                f.write("\t".join(row) + "\n")
        print(f"\n[Predict] Results saved to TSV: {args.output}")


if __name__ == '__main__':
    main()

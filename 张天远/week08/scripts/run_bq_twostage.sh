#!/bin/bash
# BQ Two-stage: BiEncoder(BQ)召回 + CrossEncoder(BQ)精排
# 防污染：临时换 BQ checkpoint → 跑完恢复 AFQMC
set -e
cd /root/autodl-tmp/text_match

echo "=== Step 1: Archive AFQMC checkpoints ==="
cp outputs/checkpoints/biencoder_cosine_best.pt outputs/checkpoints/_tmp_afqmc_bi.pt
cp outputs/checkpoints/crossencoder_best.pt       outputs/checkpoints/_tmp_afqmc_ce.pt
echo "  AFQMC archived ok"

echo "=== Step 2: Swap in BQ checkpoints ==="
cp outputs/checkpoints/biencoder_cosine_bq_corpus_best.pt outputs/checkpoints/biencoder_cosine_best.pt
cp outputs/checkpoints/crossencoder_bq_corpus_best.pt   outputs/checkpoints/crossencoder_best.pt
echo "  BQ checkpoints active"

echo "=== Step 3: Run two-stage on BQ ==="
/root/miniconda3/bin/python src/two_stage_retrieval.py --k 100 --data_dir data/bq_corpus

echo "=== Step 4: BQ two-stage result saved automatically ==="
# two_stage_retrieval.py writes directly to outputs/logs/two_stage_bq_corpus.json

echo "=== Step 5: Restore AFQMC checkpoints ==="
cp outputs/checkpoints/_tmp_afqmc_bi.pt outputs/checkpoints/biencoder_cosine_best.pt
cp outputs/checkpoints/_tmp_afqmc_ce.pt outputs/checkpoints/crossencoder_best.pt
rm outputs/checkpoints/_tmp_afqmc_bi.pt outputs/checkpoints/_tmp_afqmc_ce.pt
echo "  AFQMC restored ok"

echo ""
echo "=== DONE - BQ Two-stage result ==="
python3 -c "
import json
with open('outputs/logs/two_stage_bq_corpus.json') as f:
    data = json.load(f)
pc = data['pair_classification']
rt = data['retrieval']
print(f'  ── Pair Classification ──')
print(f'  BiEncoder     : F1={pc[\"biencoder\"][\"f1_weighted\"]:.4f}')
print(f'  CrossEncoder  : F1={pc[\"crossencoder\"][\"f1_weighted\"]:.4f}')
print(f'  ── Retrieval ──')
print(f'  BiEncoder MRR : {rt[\"biencoder\"][\"mrr\"]:.4f}')
print(f'  Bi+CE MRR     : {rt[\"crossencoder_rerank\"][\"mrr\"]:.4f}')
print(f'  Bi  Recall@10 : {rt[\"biencoder\"][\"recall_at_k\"][\"10\"]:.4f}')
print(f'  Bi+CE Recall@10: {rt[\"crossencoder_rerank\"][\"recall_at_k\"][\"10\"]:.4f}')
"

import numpy as np
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

def read_fasta(file_path):
    """读取FASTA文件并返回序列列表"""
    return [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]

def calculate_similarity(seq1, seq2, aligner):
    """计算两个序列的相似性（身份百分比）"""
    alignments = aligner.align(seq1, seq2)
    max_alignment = max(alignments, key=lambda x: x.score)
    identity = max_alignment.score / min(len(seq1), len(seq2))  # 身份百分比
    return identity

def compute_similarity_distribution(sequences1, sequences2, aligner):
    """计算两个序列集合的相似性分布"""
    similarities = []
    for seq1 in sequences1:
        for seq2 in sequences2:
            similarity = calculate_similarity(seq1, seq2, aligner)
            similarities.append(similarity)
    return similarities

# 初始化PairwiseAligner
aligner = PairwiseAligner()
aligner.mode = 'global'
aligner.match_score = 1
aligner.mismatch_score = 0

# 读取FA文件
afp_sequences = read_fasta("antifu.fa")
non_afp_sequences = read_fasta("no.fa")

# 计算相似性分布
similarity_afp_afp = compute_similarity_distribution(afp_sequences, afp_sequences, aligner)
similarity_afp_nonafp = compute_similarity_distribution(afp_sequences, non_afp_sequences, aligner)

# 绘制相似性分布图
plt.figure(figsize=(10, 6))
sns.histplot(similarity_afp_afp, color='blue', label='afps vs afps', kde=True)
sns.histplot(similarity_afp_nonafp, color='red', label='afps vs non-afps', kde=True)
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.title('Sequence Similarity Distributions')
plt.legend()
plt.show()

# 进行Wilcoxon检验
# 注意：确保两个相似性分布长度相同，否则wilcoxon会报错
min_length = min(len(similarity_afp_afp), len(similarity_afp_nonafp))
stat, p_value = wilcoxon(similarity_afp_afp[:min_length], similarity_afp_nonafp[:min_length])
print(f'Wilcoxon test statistic: {stat}, p-value: {p_value}')

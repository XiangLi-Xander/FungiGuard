import numpy as np
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Align import AlignInfo
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

def read_fasta(file_path):
    """读取FASTA文件并返回序列列表"""
    return [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]

def calculate_similarity(seq1, seq2):
    """计算两个序列的相似性（身份百分比）"""
    alignment = MultipleSeqAlignment([seq1, seq2])
    summary_align = AlignInfo.SummaryInfo(alignment)
    return summary_align.dumb_consensus(threshold=0.5, ambiguous='N')

def compute_similarity_distribution(sequences1, sequences2):
    """计算两个序列集合的相似性分布"""
    similarities = []
    for seq1 in sequences1:
        for seq2 in sequences2:
            similarity = calculate_similarity(seq1, seq2)
            similarities.append(similarity)
    return similarities

# 读取FA文件
afp_sequences = read_fasta("AFP.fa")
non_afp_sequences = read_fasta("no.fa")

# 计算相似性分布
similarity_afp_afp = compute_similarity_distribution(afp_sequences, afp_sequences)
similarity_afp_nonafp = compute_similarity_distribution(afp_sequences, non_afp_sequences)

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
stat, p_value = wilcoxon(similarity_afp_afp, similarity_afp_nonafp)
print(f'Wilcoxon test statistic: {stat}, p-value: {p_value}')

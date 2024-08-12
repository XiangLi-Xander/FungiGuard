# randomly generate protein sequences
# 随机生成蛋白质
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import random
import string

def generate_random_sequence(min_length, max_length, seed=None):
    random.seed(seed)
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    length = random.randint(min_length, max_length)
    sequence = ''.join(random.choice(amino_acids) for _ in range(length - 1))  # 生成长度为length-1的随机序列
    return 'M' + sequence  # 在开头添加M

def generate_protein_sequences(num_sequences, min_length, max_length):
    protein_sequences = []
    for i in range(num_sequences):
        length = random.randint(min_length, max_length)
        sequence = generate_random_sequence(min_length, max_length, seed=i)
        seq_record = SeqRecord(Seq(sequence), id=f'protein_{i}', description=f'Protein sequence {i} of length {length}')
        protein_sequences.append(seq_record)
    return protein_sequences

def main():
    num_sequences = 10000  # 指定生成的蛋白质序列数量
    min_length = 30  # 指定每个序列的最小长度
    max_length = 50  # 指定每个序列的最大长度
    protein_sequences = generate_protein_sequences(num_sequences, min_length, max_length)
    SeqIO.write(protein_sequences, 'random_protein_sequences.fa', 'fasta')

if __name__ == "__main__":
    main()
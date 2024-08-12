# alanine scanning
# 丙氨酸扫描
def mutate_to_alanine(protein_sequence):
    mutated_sequences = []
    for i in range(len(protein_sequence)):
        mutated_sequence = list(protein_sequence)
        mutated_sequence[i] = 'A'
        mutated_sequences.append("".join(mutated_sequence))
    return mutated_sequences

def read_fasta_file(file_path):
    sequences = []
    with open(file_path, "r") as f:
        sequence = ""
        for line in f:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return sequences

# 从FASTA文件中读取蛋白质序列
fasta_file = "input.fa"
protein_sequences = read_fasta_file(fasta_file)

# 对每个蛋白质序列依次进行突变操作并输出到FASTA文件
output_file = "mutated_to_alanine.fa"
with open(output_file, "w") as f:
    for seq_num, seq in enumerate(protein_sequences, start=1):
        mutated_sequences = mutate_to_alanine(seq)
        for i, mutated_seq in enumerate(mutated_sequences):
            f.write(">Sequence {} Mutation {}\n{}\n".format(seq_num, i+1, mutated_seq))

print("已将每个氨基酸依次突变为丙氨酸，并保存到 mutated_to_alanine.fa 文件中。")
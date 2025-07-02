# FungiGuard
FungiGuard is an innovative project aimed at developing an artificial intelligence solution for the classification of plant antifungal peptides (AFPs). Leveraging machine learning techniques, including Random Forest (RF), Long Short-Term Memory (LSTM), and attention mechanisms.

## Features

- Read protein sequences from a `.fa` file.
- Process sequences to ensure they are of the correct length and format.
- Classify sequences using five different pre-trained models.
- Output classification results to an Excel file.

## Prerequisites

To run the script, you need to have the following software installed:

- Python 3.6 or higher
- Required Python packages (listed below)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/XiangLi-Xander/FungiGuard.git
    cd FungiGuard
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    conda create -n fungiguard python=3.8
    conda activate fungiguard
    ```

3. **Install required packages:**

    Install the required Python packages using `pip`. You can install them from a `requirements.txt` file or manually:

    ```bash
    pip install -r requirements.txt
    ```

# FungiGuard Prediction Tool

## Usage

1. **Prepare your `.fa` file:**  
   Ensure your protein sequences are in FASTA format (`.fa` or `.fasta`).  
   If you want to update or change the training dataset, please modify the sequence and label data in `FungiGuard/data/antifu.xlsx` and `FungiGuard/data/no.xlsx`.

2. **Run the prediction script:**  
   Execute the Python script from the command line, specifying your `.fa` file and the trained model you want to use.

    ```bash
    cd script
    python new_peps_classifier.py path/to/protein_sequences.fa --model path/to/model_file --output path/to/output.csv
    ```

    - Replace `path/to/protein_sequences.fa` with your FASTA file path (e.g., `../demo/antifu.fa`).  
    - Replace `path/to/model_file` with one of the trained model files:  
      - Random Forest: `../models/rf_model.pkl`  
      - LSTM: `../models/lstm.pth`  
      - LSTM + Attention: `../models/lstmatt.pth`  
      - BiLSTM: `../models/bilstm.pth`  
      - BiLSTM + Attention: `../models/bilstmatt.pth`  
    - Replace `path/to/output.csv` with the desired output CSV file path (default is `predictions.csv`).

3. **Optional parameters:**  
   You can specify the maximum sequence length (default 100) with `--max_len`:

    ```bash
    python new_peps_classifier.py  ../demo/antifu.fa --model ../models/rf_model.pkl --output rf_result.csv  --max_len 100
    python new_peps_classifier.py  ../demo/antifu.fa --model ../models/lstm.pth --output lstm_result.csv  --max_len 100
    python new_peps_classifier.py  ../demo/antifu.fa --model ../models/lstmatt.pth --output lstmatt_result.csv  --max_len 100
    python new_peps_classifier.py  ../demo/antifu.fa --model ../models/bilstm.pth --output bilstm_result.csv  --max_len 100
    python new_peps_classifier.py  ../demo/antifu.fa --model ../models/bilstmatt.pth --output bilstmatt_result.csv  --max_len 100
    ```

4. **Output:**  
   The prediction results will be saved as a CSV file at the specified output path. The CSV contains:  
   - Sequence IDs (from the FASTA headers)  
   - Predicted class labels (e.g., 0 or 1)  
   - Prediction probabilities for the positive class

## Model Files

Ensure that the following model files are present in the repository directory:

- `lstm.pth`
- `lstmatt.pth`
- `bilstm.pth`
- `biLSTMATT.pth`
- `rf_model.pkl`

These models should be pre-trained and saved using PyTorch.

## Error Handling

- Sequences longer than 100 amino acids will raise an error and terminate the script.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact [1365697070@qq.com].

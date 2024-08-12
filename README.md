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
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install required packages:**

    Install the required Python packages using `pip`. You can install them from a `requirements.txt` file or manually:

    ```bash
    pip install -r requirements.txt
    ```

    If you don't have a `requirements.txt`, manually install the following packages:

    ```bash
    pip install numpy pandas biopython torch
    ```

## Usage

1. **Prepare your `.fa` file:** Ensure your protein sequences are in a `.fa` file format.

2. **Run the script:** Execute the Python script from the command line, providing the path to your `.fa` file.

    ```bash
    python new_peps_classifier.py path/to/protein_sequences.fa
    ```

    Replace `path/to/protein_sequences.fa` with the path to your `.fa` file.

3. **Output:** The results will be saved in an Excel file named `prediction_results.xlsx` in the `data` directory. Each sheet in the Excel file corresponds to a different model, showing the sequence, predicted class, and probability.

## Model Files

Ensure that the following model files are present in the repository directory:

- `model1.pki`
- `model2.pki`
- `model3.pki`
- `model4.pki`
- `model5.pki`

These models should be pre-trained and saved using PyTorch.

## Error Handling

- Sequences longer than 100 amino acids will raise an error and terminate the script.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact [1365697070@qq.com].

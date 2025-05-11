# DLF-Lite

## Overview
DLF-Lite is a **Multi-Modal NLP** project designed to explore and implement advanced techniques in Natural Language Processing (NLP). This project integrates multiple modalities (e.g., text, images, or other data types) to enhance the understanding and processing of natural language.

## Features
- Multi-modal data integration for NLP tasks.
- Advanced NLP techniques for text analysis and understanding.
- Scalable and modular architecture for experimentation.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/hithuv/DLF-Lite.git
   cd dlf-lite
   ```
2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Downloading the Dataset

To download and prepare the dataset for the project, there are 2 ways:

1. Download directly from google drive
2. Download and process using CMU-MultimodalSDK

### Downloading from google drive

1. Simply run `data/get_data.py` file
    ```bash
    cd data
    python get_data.py
    ```

2. Download directly from google drive using [link](https://drive.google.com/drive/u/1/folders/1BTFoX4LmaFdA6ikGZcj8KNh7DyDHFQCH) and place the `aligned_mosei_dataset.pkl` to `data/` folder

### Download and setup from CMU-MultimodalSDK

1. **Create a `data/` directory**:
   Ensure you have a directory named `data/` in the project root. If it doesn't exist, create it:
   ```bash
   mkdir data
   ```
2. **Download and install CMU-MultimodalSDK**:
    Clone the CMU-MultimodalSDK repository into the data/ folder:
    ```bash
    cd data
    git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
    cd CMU-MultimodalSDK
    pip install -e .
    cd ..
    ```

3. **Run the `process_mosei` script to download and process the data**:
    Use the `process_mosei` script to download and process the MOSEI dataset. This will create the following directories:
    - `cmumosei_highlevel`
    - `cmumosei_labels`
    - `cmumosei_raw`
    - `word_aligned_highlevel`
    - `final_aligned` (this contains the final processed data)
    
    ```bash
    python process_mosei.py
    ```

4. **Convert the final data to `.npy` format**:
    Use the export_from_csd script to convert the final_aligned data into a .pkl file:
    ```bash
    python export_from_csd.py
    ```

5. **Verify the final data**:
    The final `.pkl` file will be stored in the `data/` directory. Ensure the file is present.

Now the dataset is ready for use in the project.

## Running the Project

To train and evaluate the models, use the `run.py` script. This script allows you to run different models (Baseline 1, Baseline 2, Improved 1, and Improved 2) by specifying the appropriate flags.

### Steps to Run:

1. **Ensure the dataset is ready**:
   - Make sure the dataset (`aligned_mosei_dataset.pkl`) is in the `data/` directory. Refer to the [Downloading the Dataset](#downloading-the-dataset) section for instructions. 
   - Make sure to create `models/` and `csv/` directories are created in the root before running the models

2. **Run the script**:
   - Use the following commands to run specific models:

        #### Baseline 1: Late Fusion Transformer
        ```bash
        python run.py --run1
        ```

        #### Baseline 2: Late Fusion with Cross-Modal Attention
        ```bash
        python run.py --run2
        ```

        #### Improved 1: Late Fusion with Orthogonality
        ```bash
        python run.py --run3
        ```

        #### Improved 2: Late Fusion with Auxiliary Heads
        ```bash
        python run.py --run4
        ```

        #### All models
        ```bash
        python run.py --run1 --run2 --run3 --run4
        ```

3. **Configuration**:

    - The default configuration file is `config.json`. You can modify it to adjust hyperparameters like learning rate, batch size, number of epochs, etc.
    - To specify a custom configuration file, use the --config flag:

        ```bash
        python run.py --config custom_config.json --run1
        ```

4. **Output**:

    - Model checkpoints will be saved in the `models/` directory.
    - Metrics (loss, accuracy, etc.) for each epoch will be saved as CSV files in the `csv/` directory.

## Project Structure

The project is organized as follows:

```
DLF-Lite/
├── data/                   # Dataset and data processing scripts
│   ├── aligned_mosei_dataset.pkl
│   ├── get_data.py
│   └── CMU-MultimodalSDK/
├── models/                 # Saved model checkpoints
│   ├── baseline1.pth
│   ├── baseline2.pth
│   ├── improved_ortho.pth
│   └── improved_aux.pth
├── csv/                    # Metrics saved as CSV files
│   ├── baseline1_metrics.csv
│   ├── baseline2_metrics.csv
│   ├── ortho_metrics.csv
│   └── aux_metrics.csv
├── config.json             # Default configuration file
├── run.py                  # Main script to train and evaluate models
├── train.py                # Training functions
├── eval.py                 # Evaluation functions
├── utils.py                # Utility functions
├── model_baseline1.py      # Baseline 1 model
├── model_baseline2.py      # Baseline 2 model
├── improved_ortho.py       # Improved 1 model
└── improved_aux.py         # Improved 2 model
```


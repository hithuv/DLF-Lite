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
   git clone https://github.com/your-username/dlf-lite.git
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

2. Download directly from google drive using [link](https://drive.google.com/drive/u/1/folders/1BTFoX4LmaFdA6ikGZcj8KNh7DyDHFQCH) and place the `aligned_mosei_dataset.npy` to `data/` folder

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
    Use the export_from_csd script to convert the final_aligned data into a .npy file:
    ```bash
    python export_from_csd.py
    ```

5. **Verify the final data**:
    The final `.npy` file will be stored in the `data/` directory. Ensure the file is present.

Now the dataset is ready for use in the project.
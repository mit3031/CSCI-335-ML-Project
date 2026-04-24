# Sentiment Analysis on Twitter Entity Data

## Project Team
| Name                 |
|:---------------------|
| **Marvynn Talusan** |
| **Angel Ramash** |
| **Terry Huang** |
| **Prajnyique Ghimire** |

---

## Project Overview
This project builds and compares classical machine learning algorithms with modern transformer based neural networks for sentiment classification on tweets, using the Twitter Entity Sentiment Analysis dataset

---

## Environment Requirements

### Required Software
- Python **3.8+** - Git

Verify Python version:

```bash
python --version
pip --version
```

## Running the Project

From the project root directory, install the required dependencies:

```bash
pip install -r requirements.txt
```

To run the Exploratory Data Analysis and generate the dataset distribution charts:

```bash
python code/eda.py
```

To train the classical baseline models, evaluate their performance, and generate the confusion matrices:

```bash
python code/baseline_models.py
```
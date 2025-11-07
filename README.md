# DMY_Sqrt

## 🎯 Overview  
DMY_Sqrt is an AI‑model‑based project developed for school usage. It includes a trained model capable of handling square‑root (and potentially other math) problems, along with data‑generation and prediction scripts.

## 📁 Repository Contents  
Here’s what you’ll find in this repo:  
- `data/` — data folder (raw/generated data).  
- `modelDMYTest/` — model folder (contains pretrained `.pt` files).  
- `.directory` — project metadata.  
- `.gitignore` — list of files/folders ignored by Git.  
- `DMY_MathTrainer.pt` — pretrained model file for general math training.  
- `DMYsqrt.pt` — pretrained model file specifically for square‑root problems.  
- `accuracy_vs_digits.png` — performance/accuracy visualization.  
- `addition_problems.csv` — sample data (addition operation) for demonstration.  
- `data_setMaker.py` — script to generate datasets for training.  
- `model_test.py` — script to test the model on example problems.  
- `predict.py` — script to make predictions using the trained model.  
- `problems.txt` — list of sample problems/questions.  
- `trainer.py` — training script to (re)train the model from data.

## 🚀 Quick Start  
### Prerequisites  
- Python 3.x installed.  
- Recommended: create a virtual environment to isolate dependencies.

### Usage  
1. To generate training data:
   
`python data_setMaker.py`

2. To train or retrain the model:  

`python trainer.py`

3. To test the pretrained or newly trained model on example problems:  

`python model_test.py`

4. To make predictions on custom problems:  

`python predict.py`

## Performance  
- The repository contains `accuracy_vs_digits.png` illustrating model accuracy relative to problem complexity.

## Contributing  
Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests.

## License  
No license specified. Contact the repository owner for permissions.

## Contact  
Maintained by ByReals. Reach out via GitHub for questions or collaboration.

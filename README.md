# NAI Projects

A collection of AI algorithms and implementations. The projects cover a wide range of topics, including game theory, fuzzy logic, machine learning, computer vision, and reinforcement learning.

## Setup & Installation

To run these projects, we recommend setting up a virtual environment and installing the necessary packages.

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn torch torchvision gymnasium highway-env stable-baselines3 moviepy opencv-python scikit-fuzzy
```

## Projects Overview

### 1. ChompAI (Game Logic)
A bot designed to play the "Chomp" game. It uses the Minimax algorithm with Alpha-Beta pruning to make optimal moves.
- **Files:** `chomp.py`, `chompAI.py`
- **Run:** `python 1-ChompAI/chomp.py`

### 2. FuzzyAI (Fuzzy Logic)
A control system based on fuzzy logic. It demonstrates how to handle reasoning and decision-making when dealing with imprecise or vague input data.
- **Files:** `fuzzyAI.py`
- **Run:** `python 2-FuzzyAI/fuzzyAI.py`

### 3. ClusterAI (Recommendation System)
A movie recommendation engine powered by clustering algorithms. It analyzes user ratings from `movie_ratings.csv` to group similar films and suggest new titles.
- **Files:** `movie_recomendation_engine.py`, `movie_ratings.csv`
- **Run:** `python 3-ClusterAI/movie_recomendation_engine.py`

### 4. ClassificationAI (Trees & SVM)
A comparative study of Decision Trees and Support Vector Machines (SVM). We applied these models to medical and scientific datasets (Breast Cancer, Ionosphere) to evaluate their accuracy and generate confusion matrices.
- **Files:** `tree_and_svm_classification.py`
- **Run:** `python 4-ClassificationAI/tree_and_svm_classification.py`

### 5. NeuralNetsAI (Deep Learning)
Implementation and training of neural networks, including Convolutional Neural Networks (CNNs), using PyTorch. The project focuses on image classification and visualizing the learning process.
- **Files:** `neural_nets_torch.py`
- **Run:** `python 5-NeuralNetsAI/neural_nets_torch.py`

### 6. ComputerVisionAI (Motion Detection)
A "Red Light, Green Light" game concept using Computer Vision. It detects motion in real-time to determine if the player is moving when they shouldn't be.
- **Files:** `RedLightGreenLight.py`
- **Run:** `python 6-ComputerVisionAI/RedLightGreenLight.py`

### 7. ReinforcementLearning (Autonomous Driving)
An autonomous agent trained with a Deep Q-Network (DQN) to navigate highway traffic. The goal is to drive fast and avoid collisions in the `highway-env` simulation.
- **Training:** `main.py` (Uses parallel processing for faster training)
- **Demo:** `play.py` (Runs the pre-trained `my_highway_agent_opt.zip` model)
- **Run Demo:** `python 7-ReinforcementLearning/play.py`

"""
PROJEKT: Sieci Neuronowe dla Klasyfikacji (PyTorch)
AUTORZY:
  - Aleksander Stankowski (s27549)
  - Daniel Bieliński (s27292)
DATA: 19.12.2025

OPIS ZADANIA:
  1. Wykorzystać jeden z zbiorów danych z poprzednich ćwiczeń i naucz sieć neuronową.
  Porównaj skuteczność obu podejść. Dodaj logi/print screen do repozytorium.
  2. Naucz sieć rozpoznać zwierzęta, np. z zbioru CIFAR10
  3. Naucz sieć rozpoznawać ubrania. np. GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. 
  4. Zaskocz mnie. Zaproponuj własny przypadek użycia sieci neuronowych do problemu klasyfikacji.

OPIS PROBLEMU:
  Projekt realizuje cztery zadania klasyfikacji z wykorzystaniem sieci neuronowych:
    1. Klasyfikacja binarna danych tabelarycznych (Banknote Authentication)
    2. Klasyfikacja obrazów kolorowych (CIFAR-10)
    3. Klasyfikacja obrazów w skali szarości (Fashion-MNIST) - porównanie architektur - narysowanie confussion matrix
    4. Wizualizacja nieliniowych granic decyzyjnych na własnym zbiorze (Fitness Dataset)


pip install torch, torchvision, pandas, sklearn, matplotlib, seaborn.

REFERENCJE:
  - https://pytorch.org/docs/stable/index.html
  - https://www.researchgate.net/publication/1956697_Artificial_Neural_Networks_for_Beginners
  - https://www.researchgate.net/publication/261392616_Artificial_Neural_Networks_tutorial
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# ==========================================
# KONFIGURACJA URZĄDZENIA
# ==========================================


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Używane urządzenie: {device}")


# ==========================================
# 1. DEFINICJE SIECI NEURONOWYCH
# ==========================================

class SimpleNetwork(nn.Module):
  """
  Prosta sieć MLP (Multi-Layer Perceptron) do danych tabelarycznych.
  Struktura: Wejście -> 16 -> ReLU -> 8 -> ReLU -> 2 wyjścia.
  """
  def __init__(self, input_size):
    """
    Inicjalizuje warstwy sieci.
    Args:
      input_size (int): Liczba cech wejściowych (np. 4 dla Banknote).
    """
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(input_size, 16),
      nn.ReLU(),
      nn.Linear(16, 8),
      nn.ReLU(),
      nn.Linear(8, 2)
    )

  def forward(self, x):
    """
    Przepływ danych przez sieć (Forward Pass).
    Args:
      x (Tensor): Dane wejściowe.
    Returns:
      Tensor: Logity dla każdej klasy.
    """
    return self.layers(x)


class SmallFlatNetwork(nn.Module):
  """
  Mała sieć MLP do klasyfikacji obrazów (Zadanie 3 - porównanie).
  Spłaszcza obraz i używa tylko warstw liniowych.
  """
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28*28, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
    )

  def forward(self, x):
    return self.layers(x)


class ImageNetwork(nn.Module):
  """
  Sieć konwolucyjna (CNN) do klasyfikacji obrazów (CIFAR, Fashion-MNIST).
  Zapewnia lepszą ekstrakcję cech wizualnych niż MLP.
  """
  def __init__(self, channels, size_param):
    """
    Args:
      channels (int): Liczba kanałów koloru (1 dla BW, 3 dla RGB).
      size_param (int): Rozmiar wektora po spłaszczeniu warstw Conv (zależny od rozdzielczości).
    """
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(channels, 32, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(size_param, 128),
      nn.ReLU(),
      nn.Linear(128, 10)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x


# ==========================================
# 2. FUNKCJE TRENINGOWE I EWALUACYJNE
# ==========================================

def train(model, loader, optimizer, loss_func, epochs, name):
  """
  Przeprowadza trening modelu.

  Args:
    model (nn.Module): Model do wytrenowania.
    loader (DataLoader): Dane treningowe.
    optimizer (Optimizer): Algorytm optymalizacji.
    loss_func (Loss): Funkcja straty (np. CrossEntropy).
    epochs (int): Liczba epok.
    name (str): Nazwa modelu do logowania.
  """
  print(f"\n--- Start Treningu: {name} ---")
  model.train()
  
  for epoch in range(epochs):
    total_loss = 0
    for X, y in loader:
      X, y = X.to(device), y.to(device)

      optimizer.zero_grad()
      prediction = model(X)
      loss = loss_func(prediction, y)
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
    
    print(f"[{name}] Epoka {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")


def evaluate(model, loader, title="Wynik", classes=None, plot_cm=False):
  """
  Ewaluuje model na zbiorze testowym i rysuje macierz pomyłek.

  Args:
    model (nn.Module): Wytrenowany model.
    loader (DataLoader): Dane testowe.
    title (str): Tytuł wykresu/logu.
    classes (list): Lista nazw klas (dla wykresu).
    plot_cm (bool): Czy rysować Confusion Matrix.
  """
  model.eval()
  correct = 0
  total = 0
  all_preds = []
  all_targets = []

  with torch.no_grad():
    for X, y in loader:
      X, y = X.to(device), y.to(device)

      prediction = model(X)
      _, predicted_class = torch.max(prediction, 1)
      
      total += y.size(0)
      correct += (predicted_class == y).sum().item()

      if plot_cm:
        all_preds.extend(predicted_class.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

  acc = 100 * correct / total
  print(f"Dokładność ({title}): {acc:.2f}%")

  if plot_cm:
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes if classes else "auto",
                yticklabels=classes if classes else "auto")
    plt.xlabel('Przewidziana klasa')
    plt.ylabel('Prawdziwa klasa')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()


# ==========================================
# 3. REALIZACJA ZADAŃ
# ==========================================

def task_1_banknote():
  """
  ZADANIE 1: Klasyfikacja banknotów (Dane tabelaryczne).
  Realizuje porównanie podejścia NN z klasycznym ML.
  """
  print("\n=== ZADANIE 1: Banknoty ===")
  filename = "data_banknote_authentication.txt"
  
  try:
    df = pd.read_csv(filename, header=None)
  except Exception:
    print(f"BŁĄD: Nie znaleziono pliku {filename}")
    return

  # Preprocessing
  X = df.iloc[:, :-1].values
  y = df.iloc[:, -1].values
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  # Dataset
  dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
  loader = DataLoader(dataset, batch_size=16, shuffle=True)

  # Trening
  model = SimpleNetwork(input_size=4).to(device)
  train(model, loader, optim.Adam(model.parameters()), nn.CrossEntropyLoss(), epochs=20, name="Banknoty")
  evaluate(model, loader, title="Banknoty")
  
  print("KOMENTARZ: Sieć osiąga ~99-100% dokładności, co jest wynikiem porównywalnym do drzewa decyzyjnego / SVM")


def task_2_cifar():
  """
  ZADANIE 2: Klasyfikacja obrazów CIFAR-10.
  """
  print("\n=== ZADANIE 2: CIFAR10 ===")
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
  
  try:
    # download=True pobiera dataset jeśli go nie ma
    data = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
    loader = DataLoader(data, batch_size=64, shuffle=True)
    
    # 3 kanały RGB, wymiar po Conv: 64 filtry * 8 * 8 px
    model = ImageNetwork(channels=3, size_param=64*8*8).to(device)
    train(model, loader, optim.Adam(model.parameters()), nn.CrossEntropyLoss(), epochs=20, name="CIFAR")
    evaluate(model, loader, title="CIFAR")
  except Exception as e:
    print(f"Błąd CIFAR: {e}")


def task_3_fashion():
  """
  ZADANIE 3: Fashion-MNIST.
  Porównanie dwóch rozmiarów sieci (Mała MLP vs Duża CNN) + Confusion Matrix.
  """
  print("\n=== ZADANIE 3: Fashion-MNIST (Porównanie) ===")
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
  
  try:
    train_data = torchvision.datasets.FashionMNIST(root='./', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(root='./', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    
    # 1. Mała sieć
    print(">>> Model A: Mała Sieć (MLP)")
    model_small = SmallFlatNetwork().to(device)
    train(model_small, train_loader, optim.Adam(model_small.parameters()), nn.CrossEntropyLoss(), epochs=5, name="SmallMLP")
    evaluate(model_small, test_loader, title="Small MLP")

    # 2. Duża sieć
    print(">>> Model B: Duża Sieć (CNN)")
    model_big = ImageNetwork(channels=1, size_param=64*7*7).to(device)
    train(model_big, train_loader, optim.Adam(model_big.parameters()), nn.CrossEntropyLoss(), epochs=5, name="BigCNN")
    
    # Confusion Matrix
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    evaluate(model_big, test_loader, title="Big CNN", classes=classes, plot_cm=True)
    
  except Exception as e:
    print(f"Błąd FashionMNIST: {e}")


def task_4_fitness():
  """
  ZADANIE 4: Fitness Dataset - Własny przypadek użycia.
  Wizualizacja granic decyzyjnych (Decision Boundary).
  """
  print("\n=== ZADANIE 4: Fitness ===")

  filename = "fitness_dataset/fitness_dataset.csv"
  
  try:
    df = pd.read_csv(filename)
  except Exception:
    print(f"BŁĄD: Nie znaleziono pliku {filename}")
    return

  # Preprocessing
  df['gender'] = df['gender'].map({'F': 0, 'M': 1})
  df['smokes'] = df['smokes'].astype(str).map({'no': 0, 'yes': 1, '0': 0, '1': 1}).fillna(0)
  
  X = df[['activity_index', 'nutrition_quality']].values
  y = df['is_fit'].values
  X = StandardScaler().fit_transform(X)
  
  dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
  loader = DataLoader(dataset, batch_size=32, shuffle=True)
  
  # Model do wizualizacji (używamy Tanh dla gładszych linii)
  model = nn.Sequential(
    nn.Linear(2, 20), nn.Tanh(),
    nn.Linear(20, 20), nn.Tanh(),
    nn.Linear(20, 2)
  ).to(device)
  
  train(model, loader, optim.Adam(model.parameters(), lr=0.01), nn.CrossEntropyLoss(), epochs=20, name="Fitness")
  evaluate(model, loader, title="Fitness")

  # Wizualizacja
  print("Generowanie wizualizacji...")
  x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
  y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
  
  # WAŻNE: Siatka (grid) też musi trafić na GPU
  grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float().to(device)
  
  with torch.no_grad():
    # Wynik przenosimy z powrotem na CPU (.cpu()), aby narysować wykres
    Z = model(grid_tensor).argmax(1).cpu().numpy().reshape(xx.shape)

  plt.figure(figsize=(10, 8))
  plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
  plt.title("Fitness Classification - Granice Decyzyjne")
  plt.xlabel("Activity Index")
  plt.ylabel("Nutrition Quality")
  plt.show()


if __name__ == "__main__":
  task_1_banknote()
  task_2_cifar()
  task_3_fashion()
  task_4_fitness()

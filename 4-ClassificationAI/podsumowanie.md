# Podsumowanie i Analiza Funkcji Jądra (Kernel Functions)

Poniżej przedstawiono analizę wyników klasyfikacji dla dwóch zbiorów danych (Ionosphere i Breast Cancer) oraz wyjaśnienie skuteczności poszczególnych funkcji jądra.

## 1. Wyniki Eksperymentów

### Ionosphere (Dane radarowe)
| Kernel | Accuracy | Ocena |
|:---|:---|:---|
| **RBF** | **96.23%** | **Najlepszy**. Dane radarowe są złożone i nieliniowe. |
| Linear | 87.74% | Średni. Dane nie dają się łatwo podzielić prostą linią. |
| Sigmoid| 85.85% | Średni. Podobne działanie do liniowego w tym przypadku. |
| Poly | 70.75% | Najgorszy. Jądro wielomianowe jest niestabilne bez precyzyjnego strojenia. |

### Breast Cancer (Diagnostyka medyczna)
| Kernel | Accuracy | Ocena |
|:---|:---|:---|
| **Linear** | **97.66%** | **Najlepszy**. Dane są liniowo separowalne. |
| **RBF** | **97.66%** | **Najlepszy**. RBF potrafi dostosować się do danych liniowych. |
| Sigmoid| 96.49% | Bardzo dobry. |
| Poly | 89.47% | Słaby. Zbyt duża złożoność dla prostego problemu. |

---

## 2. Szczegółowa Analiza Typów Jąder (Dlaczego dobre/złe?)

### 1. Linear (Liniowe)
- **Dlaczego dobre?**
    - Jest najszybsze i najprostsze.
    - Działa idealnie, gdy klasy można oddzielić prostą linią (lub płaszczyzną w 3D).
    - W naszym przypadku świetnie sprawdziło się dla **Breast Cancer** (97.66%), co dowodzi, że cechy nowotworów łagodnych i złośliwych są wyraźnie oddzielone w przestrzeni cech.
- **Dlaczego złe?**
    - Nie radzi sobie z danymi, które są "przemieszane" (nieliniowe).
    - Widać to w **Ionosphere** (87.74%), gdzie przegrało z RBF o prawie 10 punktów procentowych.

### 2. RBF (Radial Basis Function - Jądro Gaussa)
- **Dlaczego dobre?**
    - To najbardziej uniwersalne jądro. Mapuje dane do nieskończenie wymiarowej przestrzeni, pozwalając na tworzenie bardzo elastycznych, "okrągłych" granic decyzyjnych.
    - Zwyciężyło w zbiorze **Ionosphere** (96.23%), ponieważ sygnały radarowe mają skomplikowaną strukturę.
- **Dlaczego złe?**
    - Może łatwo doprowadzić do przeuczenia (overfitting), jeśli parametr `gamma` jest źle dobrany (choć w naszych testach działało wyśmienicie).
    - Jest bardziej kosztowne obliczeniowo niż Linear.

### 3. Poly (Wielomianowe)
- **Dlaczego dobre?**
    - Teoretycznie potrafi modelować specyficzne krzywizny granic decyzyjnych.
- **Dlaczego złe?**
    - W naszych testach wypadło **najgorzej** (70% Ionosphere, 89% Breast Cancer).
    - Dlaczego? Jądro wielomianowe jest bardzo wrażliwe na parametry (`degree`, `coef0`). Bez zaawansowanego strojenia (GridSearch) często "szaleje" i tworzy granice, które nie pasują do danych testowych.

### 4. Sigmoid (Sigmoidalne)
- **Dlaczego dobre?**
    - Naśladuje działanie sieci neuronowych (funkcja aktywacji tangens hiperboliczny). Może być użyteczne w specyficznych przypadkach.
- **Dlaczego złe?**
    - Jest najmniej stabilne matematycznie spośród wymienionych. Często zachowuje się gorzej niż RBF i Linear, co potwierdziły nasze wyniki (zawsze plasowało się w środku lub na końcu stawki).

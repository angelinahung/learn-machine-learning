# Wine Dataset Raw Data

This folder contains the raw wine classification dataset used in the machine learning project.

## Dataset Information

- **Source**: Scikit-learn Wine Dataset  
- **Features**: 13 chemical properties of wine
- **Target**: Wine class (3 classes: Class_0, Class_1, Class_2)
- **Samples**: 178 total samples

## Dataset Description

The wine dataset is a classic and very easy multi-class classification dataset. It contains the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars.

### Features:
1. Alcohol
2. Malic acid  
3. Ash
4. Alcalinity of ash  
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

### Target Classes:
- **Class 0**: Wine Type 1 (59 samples)
- **Class 1**: Wine Type 2 (71 samples)  
- **Class 2**: Wine Type 3 (48 samples)

## Usage

The raw dataset is automatically loaded from scikit-learn in the preprocessing pipeline. This folder serves as a placeholder for the data source location in a production environment.

In a real-world scenario, you would:
1. Place your raw CSV/data files here
2. Update the automation script to read from these files
3. Ensure proper data versioning and backup

## Data Loading

The dataset is loaded using:
```python
from sklearn.datasets import load_wine
wine = load_wine()
```

For custom datasets, modify the `load_data()` function in `automate_angelina.py` to read from files in this directory.
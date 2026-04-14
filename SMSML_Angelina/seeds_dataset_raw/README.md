# Seeds Dataset Raw Data

This folder contains the raw seeds classification dataset used in the machine learning project.

## Dataset Information

- **Source**: UCI ML Repository - Seeds Dataset  
- **Features**: 7 geometric measurements of wheat kernels
- **Target**: Wheat variety (3 classes: Kama, Rosa, Canadian)
- **Samples**: 210 total samples

## Dataset Description

The seeds dataset contains measurements of geometric parameters of wheat kernels belonging to three different varieties of wheat: Kama, Rosa and Canadian. The kernel geometry is analyzed using a soft X-ray technique.

### Features:
1. Area
2. Perimeter  
3. Compactness
4. Length of kernel  
5. Width of kernel
6. Asymmetry coefficient
7. Length of kernel groove

### Target Classes:
- **Class 0**: Kama (70 samples)
- **Class 1**: Rosa (70 samples)  
- **Class 2**: Canadian (70 samples)

## Usage

The raw dataset is loaded from UCI repository in the preprocessing pipeline. This folder serves as a placeholder for the data source location in a production environment.

In a real-world scenario, you would:
1. Place your raw CSV/data files here
2. Update the automation script to read from these files
3. Ensure proper data versioning and backup

## Data Loading

The dataset is loaded using:
```python
import urllib.request
import pandas as pd

# Download from UCI repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
response = urllib.request.urlopen(url)
content = response.read().decode('utf-8')
```

For custom datasets, modify the `load_data()` function in `automate_angelina.py` to read from files in this directory.
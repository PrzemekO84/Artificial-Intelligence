import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


file_path = "plik_psi.txt"


try:
    data = np.loadtxt(file_path)
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")



x = data[:, 0]
y = data[:, 1]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Linear Model
liniowy_model = np.vstack([x_train, np.ones_like(x_train)]).T
wspol_liniowy = np.linalg.lstsq(liniowy_model, y_train, rcond=None)[0]

# Quadratic Model
kwardatowy_model = np.column_stack([x_train**3, x_train**2, x_train, np.ones_like(x_train)])
wspol_kwadratowy = np.linalg.lstsq(kwardatowy_model, y_train, rcond=None)[0]

# Generate x values for plotting
x_values = np.linspace(min(x_train), max(x_train), 100)

# Plotting
plt.figure(figsize=(9, 6))

# Scatter plot of training data
plt.scatter(x_train, y_train, color='green', label='Training')

# Linear Model Plot
plt.plot(x_values, wspol_liniowy[1] + wspol_liniowy[0] * x_values, color='red', linewidth=2, label='First Model')

# Quadratic Model Plot
plt.plot(x_values, wspol_kwadratowy[0] * x_values**4 + wspol_kwadratowy[1] * x_values**2 +
         wspol_kwadratowy[2] * x_values + wspol_kwadratowy[3], color='blue', linewidth=2, label='Second Model')

# Labeling and Title
plt.xlabel('Feature X')
plt.ylabel('Label Y')
plt.title('Linear Model')
plt.legend()
plt.tight_layout()
plt.show()
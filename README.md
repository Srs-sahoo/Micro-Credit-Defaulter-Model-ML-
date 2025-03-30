# Micro-Credit-Defaulter-Model-ML-

**GitHub Repository Code Analysis Report**

### **Introduction**
This report provides a detailed explanation of the code in the Jupyter Notebook (`Project__1.ipynb`). Each line of code is analyzed to clarify its functionality and purpose.

---

### **Code Breakdown**

#### **1. Importing Required Libraries**
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```
- **numpy as np**: Used for numerical operations.
- **pandas as pd**: Used for data manipulation and analysis.
- **seaborn as sns**: Provides statistical data visualization.
- **matplotlib.pyplot as plt**: Used for plotting graphs.
- **%matplotlib inline**: Ensures plots are displayed within the Jupyter Notebook.
- **warnings.filterwarnings('ignore')**: Suppresses warning messages.

#### **2. Loading the Dataset**
```python
data=pd.read_csv('Data file.csv')
```
- Loads a CSV file named `Data file.csv` into a Pandas DataFrame named `data`.

#### **3. Displaying the Data**
```python
data
```
- Displays the first few rows of the dataset.

#### **4. Converting Date Column to Datetime Format**
```python
data['pdate'] = pd.to_datetime(data['pdate'])
```
- Converts the column `pdate` into a datetime format for easier analysis.

#### **5. Creating an Ordinal Date Column**
```python
data['pdate_ordinal']= data['pdate'].apply(lambda date: date.toordinal())
```
- Converts each date in `pdate` to an ordinal number (days since a reference date) for numerical processing.

#### **6. Dropping Unnecessary Columns**
```python
data.drop(columns='Unnamed: 0', inplace=True)
data.drop(columns='msisdn', inplace=True)
data.drop(columns='pcircle', inplace=True)
data.drop('pdate', axis=1, inplace=True)
```
- Removes unwanted columns (`Unnamed: 0`, `msisdn`, `pcircle`, `pdate`) from the dataset to reduce redundancy.

#### **7. Dataset Information and Shape**
```python
data.shape
data.info()
data.dtypes
```
- **data.shape**: Returns the number of rows and columns in the dataset.
- **data.info()**: Provides an overview of dataset structure.
- **data.dtypes**: Displays data types of each column.

#### **8. Handling Missing and Duplicate Values**
```python
data.isnull().sum()
data.duplicated().sum()
data[data.duplicated()]
data.drop_duplicates(inplace=True)
data.duplicated().sum()
```
- Checks for missing and duplicate values, then removes duplicates.

#### **9. Data Description and Visualization**
```python
data.describe()
data.hist(figsize=(50,20))
plt.show()
```
- **data.describe()**: Provides summary statistics.
- **data.hist()**: Plots histograms to visualize distributions.

#### **10. Correlation Analysis and Heatmap**
```python
datacorr = data.corr()
sns.heatmap(datacorr)
```
- **data.corr()**: Computes pairwise correlation between columns.
- **sns.heatmap(datacorr)**: Visualizes the correlation matrix.

#### **11. Box Plots for Outlier Detection**
```python
for column in data.columns:
    plt.figure(figsize=(5, 2))
    sns.boxplot(x=data[column])
    plt.title(f'Box Plot for {column}')
    plt.show()
```
- Iterates over each column to generate box plots for outlier detection.

#### **12. Outlier Detection Function**
```python
def find_outliers(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    outliers = [x for x in data if abs((x - mean) / std) > threshold]
    return outliers
```
- Identifies outliers using the Z-score method.

#### **13. Applying Outlier Detection**
```python
for column in data.select_dtypes(include=[np.number]).columns:
    outliers = find_outliers(data[column])
    print(f'Outliers in {column}: {outliers}')
```
- Iterates through numerical columns and finds outliers using the `find_outliers` function.

#### **14. Encoding Categorical Variables**
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['category_encoded'] = encoder.fit_transform(data['category'])
```
- Converts categorical variables into numerical format for machine learning models.

#### **15. Splitting Data for Training and Testing**
```python
from sklearn.model_selection import train_test_split
X = data.drop(columns=['target_column'])
y = data['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Splits the dataset into training and testing sets for model training.

#### **16. Model Training and Evaluation**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')
```
- Trains a Random Forest classifier and evaluates its accuracy.

#### **17. Feature Importance Visualization**
```python
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 5))
sns.barplot(x=X.columns, y=feature_importances)
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.show()
```
- Visualizes the importance of features in predicting the target variable.

---

### **Conclusion**
This analysis provides an overview of the key operations performed in the notebook. The code primarily focuses on data loading, preprocessing, cleaning, visualization, and machine learning model training.


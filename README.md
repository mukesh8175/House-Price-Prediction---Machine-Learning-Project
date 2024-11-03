# House Price Prediction - Machine Learning Project

## Project Overview
This project is focused on predicting house prices in Bengaluru using a combination of machine learning techniques. It involves thorough data cleaning, preprocessing, and feature engineering to prepare the dataset for model training. four regression models, Linear Regression, Ridge Regression, Lasso Regression and Decision Tree Regression, are employed to capture both linear and non-linear relationships in the data.

## Key Features
### 1. Data Cleaning & Processing:
- **Handling Missing Values**: Numerical features with missing values were filled using mean, median, or mode, depending on the context of the data.
- **Outlier Detection and Removal**: The Interquartile Range (IQR) method was used to identify and remove outliers, leading to cleaner and more reliable data.

### 2. Feature Engineering:
- **Column Transformers**: Applied transformations like One-Hot Encoding for categorical variables and scaling for numerical features.
- **One-Hot Encoding (OHE)**: Converted categorical variables such as location and house type into numerical format for model compatibility.
- **Scaling**: Standardized numerical features to bring them to the same scale, improving the performance of certain models like Ridge and Lasso.
- **Pipelines**: Implemented a pipeline to automate the entire data preprocessing, feature engineering, and modeling process, making it efficient and reproducible.

## Machine Learning Models
- **Linear Regression**: A simple model to establish baseline performance. It captures linear relationships between house features and prices.
- **Decision Tree Regression**: A non-linear model that captures complex patterns in the data. Useful for handling both numerical and categorical features.
- **Ridge Regression**: A regularized version of linear regression that helps prevent overfitting by introducing a penalty term. Suitable for models with many correlated features.
- **Lasso Regression**: Similar to Ridge but with added feature selection capability by shrinking some feature coefficients to zero, making it a good choice for sparse data or high-dimensional datasets.

## Evaluation Metrics
To evaluate the model's performance, the following metrics were used:
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted house prices. The lower, the better.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing an interpretable error measure in the original price units.
- **R-squared (R²)**: Represents the proportion of variance in the target variable that is explained by the features. A higher R² means better explanatory power.

## Dataset
The dataset consists of various house-related features that influence house prices. Key columns include:
- **Location**: The geographical location of the house (categorical).
- **Size**: House size, usually represented by the number of bedrooms (e.g., 2 BHK).
- **Total Sqft**: Total square footage of the house.
- **Bath**: Number of bathrooms in the house.
- **Balcony**: Number of balconies.
- **Price**: The target variable representing the house price.

## Project Workflow
1. **Data Cleaning**: Handled missing values, outliers, and duplicates in the dataset.
2. **Feature Engineering**: Transformed and scaled features for better model performance.
3. **Model Training**: Applied multiple machine learning models (Linear, Decision Tree, Ridge, Lasso) and used pipelines to streamline the process.
4. **Model Evaluation**: Evaluated the performance of the models using metrics like MSE, RMSE, and R².

## Conclusion
This project successfully demonstrates the application of various regression techniques to predict house prices based on several features. By utilizing data cleaning, feature engineering, and model regularization (Ridge, Lasso), the model's accuracy and robustness were improved. Pipelines were implemented to make the workflow efficient and reproducible, ensuring smooth scaling and deployment. Further improvements could include fine-tuning hyperparameters or exploring other advanced regression models to improve prediction accuracy.


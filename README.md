# Loan Amount Prediction Machine Learning Model

The loan amount prediction model will use the Loan Dataset (https://www.kaggle.com/datasets/burak3ergun/loan-data-set) (614 samples, 13 features) to predict the loan amount a customer might qualify for based on their financial and demographic information.

I will be comparing performance between a Linear Regression model and an XGBoost model to identify the best predictor for loan amounts, generating optimal results.  

## Objective
The banking industry currently faces challenges in determining appropriate loan amounts, with traditional methods relying heavily on manual processing by loan officers. This project aims to develop a machine learning solution using regression algorithms to automate and optimize the loan amount prediction process.

**Current Industry Challenges:**
 - Manual review and calculation of loan eligibility and amount by loan officers
 - Time-consuming process
 - Inconsistent assessment criteria across different applications

**The significance of this project lies in its ability to:**  
 - Automates loan amount predictions to enable faster and more accurate lending decisions for financial institutions.  
 - Reduces risks by leveraging advanced machine learning models like Random Forest and XGBoost.  
 - Identifying complex patterns in financial data through predictive analytics.  
 - Maintain consistent evaluation criteria.  

**For the complete code, including steps for EDA, pre-preprocessing, model building, and evaluation, please refer to the Jupyter Notebook file in this repository.**  

## Exploratory Data Analysis
**Numerical Features:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term  
**Categorical Features:** Gender, Credit_History, Self_Employed, Married, Loan_Status, Education, Property_Area  
**Alphanumeric Features:** Loan_ID  

Below we can see the correlation heatmap. Applicant and Co-applicant income have the highest correlation with our target variable loan amount. We will explore the correlation heatmap further, once we create new features.  
![image](https://github.com/user-attachments/assets/825c3851-8059-4afc-bb03-f9deeab186ab)  

**Top Left (Scatter Plot: Applicant Income vs. Loan Amount):**  
This scatter plot shows that most applicants with incomes below 20,000 qualify for loans under 300. A few outliers with higher incomes or loan amounts stand apart from the main cluster.  
**Top Right (Education vs. Applicant Income):**  
The box plot highlights that graduates generally earn more than non-graduates, though both groups have outliers with very high incomes.  
**Bottom Left (Dependents vs. Loan Amount):**  
This box plot shows how loan amounts vary based on the number of dependents. Applicants with no dependents tend to apply and qualify for smaller loans, while those with 3+ dependents show a wider range of loan amounts, including higher values.  
![image](https://github.com/user-attachments/assets/ee00def4-4d2e-4d81-be0d-49bb978433b8)  

The following scatter plot helps identify how loan term preferences correlate with loan amount. This is crucial for understanding customer behaviour and predicting loan amounts based on term length.  
![image](https://github.com/user-attachments/assets/a2f658c1-5151-4ce6-ad72-155258dcbda4)  

The following bar chart helps us understand the relationship between employment type and credit history, which is critical for predicting loan eligibility, as credit history is a key factor in determining approval.  
![image](https://github.com/user-attachments/assets/73e08689-25f9-4e6a-b5da-226a5d522711)  

This last bar chart highlights how dependents affect loan approval rates, likely due to their impact on financial stability. This can help refine the model by showing how dependents influence loan status outcomes, which affects the loan amount.  
![image](https://github.com/user-attachments/assets/3c1a7ddd-faee-401f-8753-5c0169cbbaa0)  

## Data Cleaning and Preprocessing  

### Handling Missing Values (Data Imputation)
 - Missing values in categorical columns like Gender, Married, Dependents, Self_Employed, and Credit_History were replaced with their mode (most frequent value). This ensures that missing data does not disrupt the model while preserving the distribution of categories.  
 - For numerical columns like LoanAmount and Loan_Amount_Term, missing values were also replaced with their mode, as these columns often have a small number of unique values (e.g., standard loan terms like 360 months).

### Outlier Removal Using IQR (Interquartile Range)
 - Outliers in numerical columns (excluding binary-like columns such as Credit_History and Loan_Amount_Term) were identified using the IQR method.  
 - The IQR was computed as IQR = Q3 - Q1.  
 - Any data points below (Q1 - 1.5 * IQR) or above (Q3 + 1.5 * IQR) were considered outliers and removed.
 - Removing outliers ensures that extreme values do not skew the model's predictions, leading to better generalization and improved accuracy.

### Feature Engineering  
 - A new column, TotalHouseholdIncome, is created by adding the ApplicantIncome and CoapplicantIncome.  
 - Purpose: This feature represents the combined income of all contributors to the loan, which is crucial for understanding the applicant's financial capacity.  
 - A new column, MonthlyLoanPayment, is calculated by dividing the LoanAmount by the loan term (converted from months to approximate years using 30.417, the average number of days in a month).  
 - Purpose: This feature estimates the monthly payment required for the loan, which is essential for assessing affordability.  
 - A new column, DTI_Ratio, is created by dividing the MonthlyLoanPayment by the TotalHouseholdIncome.  
 - Purpose: The DTI ratio measures how much of the applicant's income goes toward debt payments, a key metric for determining loan eligibility and risk.

The new features enhance the dataset by introducing derived metrics that capture complex relationships between income, debt, and repayment terms. By improving feature diversity and relevance to the target attribute, these new features help the model better understand patterns in the data and make more accurate predictions about loan amounts.  
![image](https://github.com/user-attachments/assets/b56d1424-e360-445a-be9b-409d17073c4d)  

### One-Hot Encoding  
Machine learning models cannot process categorical data directly, so this transformation ensures that the model can interpret and utilize these features effectively.  
How One-Hot Encoding Helps:  
 - Preserves all category information without introducing ordinal relationships (e.g., one category being "greater" than another).  
 - Makes the dataset fully numerical, which is the required type of data to be fed to most machine learning algorithms.

### Addressing the Skewness  
Square Root Transformation:  
The ApplicantIncome, CoapplicantIncome, and LoanAmount columns are transformed using the square root function to reduce skewness and make their distributions closer to normal.  
Many machine learning models perform better with normally distributed data, as it helps improve model stability and predictive performance.  

Histograms with KDE Plots:  
Before addressing the skewness:  
![image](https://github.com/user-attachments/assets/725f4b5d-9dbf-46ab-8a85-68cbd4865fdc)  
After addressing the skewness:  
![image](https://github.com/user-attachments/assets/a20be682-bfba-40c7-b157-75203e75f0e4)  

## Model Development  
This section of the code focuses on building, optimizing, and evaluating two machine-learning models to predict loan amounts.  

1. Data Preparation  
 - The dataset is split into features (X) and the target variable (y), where LoanAmount is the target.  
 - A Min-Max Scaler is applied to scale the features to a range between 0 and 1, this ensures all features contribute equally to the model's performance.

2. Random Forest Regressor  
 - A Random Forest Regressor is trained as the first model I used.  
 - Random Forest Out-of-Bag Score: 0.9462111841924802 (Internal validation score)  
 - Random Forest MSE: 0.0007744609445800214 (the average of the squared differences between predicted values and actual values)  
 - Random Forest R^2: 0.9303488605074197 (explains variance in loan amounts)
 - I chose this model because it provides great predictions due to its ability to handle non-linear relationships and feature importance analysis.  

3. Hyperparameter Tuning  
 - I used grid search with a parameter grid to optimize Random Forest hyperparameters (n_estimators, max_depth).  
 - The best-tuned Random Forest I could train achieved an R² score of 0.9318908836437666 on test data.  

4. XGBoost Regressor  
 - An XGBoost Regressor is created with the parameters: objective="reg:squarederror", random_state=0, early_stopping_rounds=3
 - It was trained and evaluated, achieving an R² score of 0.9520023051777154 and an MSE score of 0.0005336932079006393, outperforming both Random Forest models.
 - I used XGBoost because it efficiently captures complex relationships in the data, leading to improved prediction accuracy.

5. K-Fold Cross-Validation (Ensures reliable performance across different data splits)  
 - Both Random Forest and XGBoost underwent 5-fold Cross-Validation.  
 - This process splits the data into five subsets to evaluate model consistency.  
 - Mean R² scores: Random Forest: 0.9468636760585474, XGBoost: 0.9488184858822706, indicating that both models performed strongly.  

## Conclusion
By comparing metrics like R², MSE, and cross-validation scores, XGBoost emerges as the better-performing model for predicting loan amounts. This process demonstrates a detailed approach to transforming raw data into accurate predictions by building a reliable machine-learning pipeline for loan amount prediction. By training, tuning, and evaluating models like Random Forest and XGBoost, the pipeline effectively predicts loan amounts with high accuracy. The integration of cross-validation and hyperparameter tuning ensures reliable performance. Overall, this project was very enjoyable to build and helped me learn a lot about the core concepts of data science and machine learning😃.  

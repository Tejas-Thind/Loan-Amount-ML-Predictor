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

Top Left (Scatter Plot: Applicant Income vs. Loan Amount):
This scatter plot shows that most applicants with incomes below 20,000 qualify for loans under 300. A few outliers with higher incomes or loan amounts stand apart from the main cluster.  
Top Right (Education vs. Applicant Income):
The box plot highlights that graduates generally earn more than non-graduates, though both groups have outliers with very high incomes.  
Bottom Left (Dependents vs. Loan Amount):
This box plot shows how loan amounts vary based on the number of dependents. Applicants with no dependents tend to apply and qualify for smaller loans, while those with 3+ dependents show a wider range of loan amounts, including higher values.  
![image](https://github.com/user-attachments/assets/ee00def4-4d2e-4d81-be0d-49bb978433b8)  

The following scatter plot helps identify how loan term preferences correlate with the loan amount, which is crucial for understanding customer behaviour and predicting loan amounts based on term length.  
![image](https://github.com/user-attachments/assets/a2f658c1-5151-4ce6-ad72-155258dcbda4)  

The following bar chart helps us understand the relationship between employment type and credit history is critical for predicting loan eligibility, as credit history is a key factor in determining approval.  
![image](https://github.com/user-attachments/assets/73e08689-25f9-4e6a-b5da-226a5d522711)  

This last bar chart highlights how dependents affect loan approval rates, likely due to their impact on financial stability. This can help refine the model by showing how dependents influence loan status outcomes, which affects the loan amount.  
![image](https://github.com/user-attachments/assets/57bd91ed-9d9e-41bd-ac16-19c06ce4dd5f)  

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


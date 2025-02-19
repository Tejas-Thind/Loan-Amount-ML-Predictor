# Loan Amount Prediction ML Model

The loan amount prediction model will use the Home Loan Prediction Dataset (614 samples, 13 features) to predict the loan amount a customer might qualify for based on common information filled in by customers while completing an online application form. 

I will be comparing performance between ____________ and XGBoost to identify the best predictor for loan amounts, generating optimal results.

## Objective
The banking industry currently faces challenges in determining appropriate loan amounts, with traditional methods relying heavily on manual processing by loan officers. This project aims to develop a machine learning solution using regression algorithms to automate and optimize the loan amount prediction process.

**Current Industry Challenges:**
Manual review and calculation of loan eligibility and amount by loan officers
Time-consuming process
Inconsistent assessment criteria across different applications

**The significance of this project lies in its ability to:**  
- Standardize loan amount calculations  
- Reduce processing time significantly  
- Allow loan officers to focus on more complex applications  
- Improve customer experience through faster decisions  
- Maintain consistent evaluation criteria

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


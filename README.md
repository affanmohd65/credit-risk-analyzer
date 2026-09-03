# India Credit Risk Analyzer

An end-to-end machine learning application for predicting retail-loan default risk in an India-focused lending portfolio.

The project demonstrates a complete credit-risk workflow: data generation, validation, feature engineering, machine-learning model comparison, probability calibration, risk scoring, FastAPI model serving, Streamlit analytics, portfolio expected-loss analysis, monitoring, testing, Docker, and GitHub Actions CI.

## Project Overview

Financial institutions need to estimate the probability that a borrower may default before approving a loan. This project predicts the probability of default for Indian retail-loan applications and converts that estimate into:

- Default probability
- Risk score
- Risk category
- Recommended decision
- Risk and protective factors

The dashboard also provides portfolio-level insights such as loan exposure, expected loss, default rate, bureau-score distribution, FOIR distribution, and risk patterns by geography and loan type.

## Business Objective

The objective is to help a lender evaluate retail-loan applications based on borrower affordability, repayment capacity, credit behavior, and existing obligations.

The application focuses on important lending signals:

- Bureau score
- Fixed Obligation to Income Ratio (FOIR)
- Existing overdue accounts
- Recent credit inquiries
- Employment type
- Annual income
- Existing EMI obligations
- Proposed EMI
- Loan amount
- Loan term
- Interest rate
- Bank balance
- Residence type
- City and loan type

## Dataset

This project uses a reproducible India-focused retail-loan portfolio generator.

The generator creates `100,000` loan applications with realistic relationships between borrower profile, financial behavior, loan attributes, and default outcomes.

### Data Characteristics

- INR-based annual income, loan amount, EMI, and bank balance
- Indian city segments including Mumbai, Delhi NCR, Bengaluru, Chennai, Hyderabad, Pune, Ahmedabad, Kolkata, and Tier 2/3 locations
- Employment types including Salaried, Self Employed, Professional, and Contract
- Loan types including Personal Loan, Two Wheeler, Consumer Durable, Home Improvement, and Business Loan
- Bureau scores from `300` to `900`
- FOIR values representing total monthly debt obligations relative to income
- Controlled missing values and outliers for data-quality validation
- Feature-dependent default generation
- Approximately 5–15% default rate

## Default-Risk Relationships

Default probability is influenced by realistic lending factors:

```text
Lower bureau score                 -> Higher default risk
Higher FOIR                        -> Higher default risk
More overdue accounts              -> Higher default risk
More recent credit inquiries       -> Higher default risk
Contract employment                -> Higher default risk
Higher loan exposure               -> Higher default risk
Lower bank balance                 -> Higher default risk
Rented residence                   -> Higher default risk
Stable employment                  -> Lower default risk
Higher bureau score                -> Lower default risk
Lower repayment burden             -> Lower default risk

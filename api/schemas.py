from pydantic import BaseModel, Field


class IndianLoanApplication(BaseModel):
    city_tier: str
    employment_type: str
    residence_type: str
    age: int = Field(ge=18, le=100)
    employment_years: float = Field(ge=0, le=50)
    annual_income_inr: float = Field(gt=0, le=100_000_000)
    bureau_score: float = Field(ge=300, le=900)
    credit_inquiries_6m: int = Field(ge=0, le=50)
    overdue_accounts: int = Field(ge=0, le=50)
    existing_emi_inr: float = Field(ge=0)
    bank_balance_inr: float = Field(ge=0)
    loan_type: str
    loan_amount_inr: float = Field(gt=0, le=20_000_000)
    loan_term_months: int = Field(ge=6, le=120)
    interest_rate: float = Field(ge=0, le=60)
    proposed_emi_inr: float = Field(gt=0)
    foir: float = Field(ge=0, le=2)
    loan_to_income_ratio: float = Field(ge=0, le=20)


class BatchRequest(BaseModel):
    applications: list[IndianLoanApplication] = Field(min_length=1, max_length=10_000)
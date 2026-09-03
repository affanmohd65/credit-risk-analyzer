from pydantic import BaseModel, Field


class CreditAccount(BaseModel):
    credit_limit: float = Field(gt=0, le=2_000_000)
    sex: int = Field(ge=1, le=2)
    education: int = Field(ge=0, le=6)
    marital_status: int = Field(ge=0, le=3)
    age: int = Field(ge=18, le=100)
    pay_status_0: int = Field(ge=-2, le=9)
    pay_status_2: int = Field(ge=-2, le=9)
    pay_status_3: int = Field(ge=-2, le=9)
    pay_status_4: int = Field(ge=-2, le=9)
    pay_status_5: int = Field(ge=-2, le=9)
    pay_status_6: int = Field(ge=-2, le=9)
    bill_amount_1: float
    bill_amount_2: float
    bill_amount_3: float
    bill_amount_4: float
    bill_amount_5: float
    bill_amount_6: float
    payment_amount_1: float = Field(ge=0)
    payment_amount_2: float = Field(ge=0)
    payment_amount_3: float = Field(ge=0)
    payment_amount_4: float = Field(ge=0)
    payment_amount_5: float = Field(ge=0)
    payment_amount_6: float = Field(ge=0)


class BatchRequest(BaseModel):
    applications: list[CreditAccount] = Field(min_length=1, max_length=10_000)
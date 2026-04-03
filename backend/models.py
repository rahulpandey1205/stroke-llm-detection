from sqlalchemy import Column, Integer, Float, String
from backend.database import Base

class PatientRecord(Base):

    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)

    age = Column(Float)
    bmi = Column(Float)
    avg_glucose_level = Column(Float)

    hypertension = Column(Integer)
    heart_disease = Column(Integer)
    smoking_status = Column(Integer)

    prediction = Column(String)
    probability = Column(Float)
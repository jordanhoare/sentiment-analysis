from sqlalchemy import Boolean, Column, Integer, Numeric, String

from .database import Base


class Phrases(Base):
    __tablename__ = "phrases"

    id = Column(Integer, primary_key=True, index=True)
    phrase = Column(String[150])
    sentiment = Column(String)
    confidence = Column(Numeric[10, 2])
    negative = Column(Numeric[10, 2])
    neutral = Column(Numeric[10, 2])
    positive = Column(Numeric[10, 2])

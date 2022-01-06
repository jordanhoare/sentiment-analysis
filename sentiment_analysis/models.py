from database import Base
from sqlalchemy import Boolean, Column, Integer, String


class Phrase(Base):
    __tablename__ = "phrases"

    id = Column(Integer, primary_key=True, index=True)
    phrase = Column(String[150])
    sentiment = Column(String)
    confidence = Column(Integer[10, 2])
    negative = Column(Integer[10, 2])
    neutral = Column(Integer[10, 2])
    positive = Column(Integer[10, 2])

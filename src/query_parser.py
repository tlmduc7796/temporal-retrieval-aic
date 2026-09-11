from dataclasses import dataclass
from typing import List

@dataclass
class Event:
    event_id: int
    text: str
    original_text: str

class QueryParser:
    def parse(self, query):
        if isinstance(query, str):
            query = [q.strip() for q in query.replace('.', ',').split(',') if q.strip()]

        events = []
        for i, text in enumerate(query):
            original = text.strip()
            expanded_text = f"A clear photo showing: {original.lower()}"
            
            events.append(
                Event(
                    event_id=i,
                    text=expanded_text,
                    original_text=original
                )
            )
        return events


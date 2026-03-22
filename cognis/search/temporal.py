"""
Temporal query parsing and decay calculations.

Ported from production cognis temporal/ modules.
"""

import re
import math
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from cognis.models import Memory


# Relative time patterns
RELATIVE_PATTERNS = [
    (r'\b(today)\b', lambda now: now),
    (r'\b(yesterday)\b', lambda now: now - timedelta(days=1)),
    (r'\b(last week)\b', lambda now: now - timedelta(weeks=1)),
    (r'\b(last month)\b', lambda now: now - timedelta(days=30)),
    (r'\b(last year)\b', lambda now: now - timedelta(days=365)),
    (r'\b(\d+)\s+days?\s+ago\b', lambda now, d: now - timedelta(days=int(d))),
    (r'\b(\d+)\s+weeks?\s+ago\b', lambda now, w: now - timedelta(weeks=int(w))),
    (r'\b(\d+)\s+months?\s+ago\b', lambda now, m: now - timedelta(days=int(m) * 30)),
]

# Absolute date patterns
ABSOLUTE_PATTERNS = [
    (r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',
     lambda y, m, d: datetime(int(y), int(m), int(d), tzinfo=timezone.utc)),
    (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
     lambda m, d, y: datetime(int(y), _month_to_num(m), int(d), tzinfo=timezone.utc)),
    (r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
     lambda d, m, y: datetime(int(y), _month_to_num(m), int(d), tzinfo=timezone.utc)),
    (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
     lambda m, y: datetime(int(y), _month_to_num(m), 15, tzinfo=timezone.utc)),
    (r'\b(?:in|year)\s+(\d{4})\b',
     lambda y: datetime(int(y), 7, 1, tzinfo=timezone.utc)),
]

TEMPORAL_KEYWORDS = [
    'when', 'what time', 'what date', 'how long ago',
    'before', 'after', 'during', 'since', 'until',
    'first', 'last', 'recent', 'latest', 'earliest',
]

# Sector-specific decay rates
SECTOR_DECAY_RATES = {
    "episodic": 0.15,
    "semantic": 0.05,
    "procedural": 0.02,
    "emotional": 0.10,
    "reflective": 0.08,
}
DEFAULT_DECAY_RATE = 0.08


def _month_to_num(month_name: str) -> int:
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
    }
    return months.get(month_name.lower(), 1)


def extract_query_date(query: str, reference_time: Optional[datetime] = None) -> Optional[datetime]:
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    # Absolute patterns first
    for pattern, converter in ABSOLUTE_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            try:
                return converter(*match.groups())
            except (ValueError, TypeError):
                continue

    # Relative patterns
    query_lower = query.lower()
    for pattern, converter in RELATIVE_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            try:
                groups = match.groups()
                if len(groups) == 1 and not groups[0].isdigit():
                    return converter(reference_time)
                elif len(groups) >= 1 and groups[0].isdigit():
                    return converter(reference_time, groups[0])
            except (ValueError, TypeError):
                continue

    return None


def is_temporal_query(query: str) -> bool:
    query_lower = query.lower()
    for keyword in TEMPORAL_KEYWORDS:
        if keyword in query_lower:
            return True
    return extract_query_date(query) is not None


def parse_temporal_query(
    query: str,
    reference_time: Optional[datetime] = None,
) -> Tuple[bool, Optional[datetime], int]:
    """
    Parse a query for temporal information.

    Returns: (is_temporal, query_date, window_days)
    """
    is_temporal = is_temporal_query(query)
    query_date = extract_query_date(query, reference_time)

    query_lower = query.lower()
    if 'today' in query_lower:
        window_days = 1
    elif 'yesterday' in query_lower:
        window_days = 2
    elif 'last week' in query_lower or 'week ago' in query_lower:
        window_days = 14
    elif 'last month' in query_lower or 'month ago' in query_lower:
        window_days = 60
    elif 'last year' in query_lower or 'year ago' in query_lower:
        window_days = 400
    else:
        window_days = 30

    return is_temporal, query_date, window_days


def calculate_temporal_relevance(
    memory: Memory,
    query_date: Optional[datetime] = None,
    window_days: int = 365,
) -> float:
    """Temporal relevance based on event_time proximity to query_date."""
    if not memory.event_time or query_date is None:
        return 1.0

    days_diff = abs((query_date - memory.event_time).days)
    if days_diff >= window_days:
        return 0.1
    return max(0.1, 1.0 - (days_diff / window_days))


def calculate_decay(memory: Memory, reference_time: Optional[datetime] = None) -> float:
    """Exponential decay: score = e^(-lambda * days)."""
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    last_time = memory.created_at
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)

    days = max(0, (reference_time - last_time).total_seconds() / 86400)
    sector = memory.metadata.sector if memory.metadata else "semantic"
    rate = SECTOR_DECAY_RATES.get(sector, DEFAULT_DECAY_RATE)
    return max(0.0, min(1.0, math.exp(-rate * days)))

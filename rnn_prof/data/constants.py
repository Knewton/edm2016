"""
Key names used in data frames passed to RNN
"""

# Represents an "item" for the RNN. Should be a continuous sequence of
# numbers in range(0, #items)
ITEM_IDX_KEY = 'item_idx'

# Represents a "template" for the RNN. Should be a continuous sequence of
# numbers in range(0, #templates)
TEMPLATE_IDX_KEY = 'template_idx'

# Represents a "concept" for the RNN. Should be a continuous sequence of
# numbers in range(0, #concepts)
CONCEPT_IDX_KEY = 'concept_idx'

# Represents a "user" for the RNN. Should be a continuous sequence of
# numbers in range(0, #users)
USER_IDX_KEY = 'user_idx'

# Represents a temporal ordering of items for the RNN. Can be any numeric
# type as it's just used for sorting.
TIME_IDX_KEY = 'time_idx'

# Whether the student got the item correct or not. Should be 0/1 or False/True
CORRECT_KEY = 'correct'

# If a data set supports hinting, the number of hints requested before answering
HINT_COUNT_KEY = 'hint_count'

# used to represent single (constant) value for concepts and templates across datasets
SINGLE = 'single'

# datasets
ASSISTMENTS = 'assistments'
KDDCUP = 'kddcup'

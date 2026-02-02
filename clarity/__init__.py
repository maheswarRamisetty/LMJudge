# The Clarity Module evaluates how clear, readable, and unambiguous a generated text (typically a summary or judgment) is, independent of factual correctness or completeness.
# Its goal is to measure linguistic quality, ensuring that information—if present—is expressed in a way that is easy to understand and difficult to misinterpret.

# The module computes clarity as a weighted combination of four interpretable components:

# Clarity
# =
# 𝛼
# 𝑅
# +
# 𝛽
# 𝑆
# +
# 𝛾
# 𝐷
# +
# 𝛿
# 𝑃
# Clarity=αR+βS+γD+δP

# where each component is normalized to 
# [
# 0
# ,
# 1
# ]
# [0,1] and all weights are equal by default

# (
# 𝛼
# =
# 𝛽
# =
# 𝛾
# =
# 𝛿
# =
# 0.25
# )
# (α=β=γ=δ=0.25).

# 1. Readability Score (R)
# Description

# Measures how easy the text is to read using the Flesch Reading Ease metric.
# Short sentences, simple vocabulary, and direct phrasing result in higher scores.

# What it captures

# Sentence length

# Word complexity

# Cognitive reading effort

# Examples

# High Readability (R ≈ 0.8–1.0)

# “The customer was charged ₹500 incorrectly. The charge was refunded.”

# Low Readability (R ≈ 0.3–0.4)

# “Pursuant to the verification of transactional inconsistencies, a financial rectification procedure was initiated.”

# 2. Syntactic Simplicity (S)
# Description

# Evaluates grammatical simplicity by measuring average dependency parse-tree depth.
# Shallower trees indicate simpler sentence structure.

# What it captures

# Clause nesting

# Sentence complexity

# Cognitive parsing effort

# Examples

# High Simplicity (S ≈ 0.8–1.0)

# “The agent reviewed the case and approved the refund.”

# Low Simplicity (S ≈ 0.4–0.5)

# “After reviewing the case that had been escalated following multiple procedural verifications, the agent approved the refund.”

# 3. Disambiguation Index (D)
# Description

# Measures whether the text contains explicit grounding details that reduce ambiguity, such as:

# Monetary values

# Dates

# Locations

# What it captures

# Explicitness

# Lack of vagueness

# Grounded facts

# Examples

# High Disambiguation (D = 1.0)

# “₹500 was charged on 12/03/2024 while the customer was in India.”

# Low Disambiguation (D ≈ 0.0–0.33)

# “An amount was charged earlier while the customer was traveling.”

# 4. Pronoun Reference Clarity (P)
# Description

# Evaluates whether pronouns have clear and unambiguous antecedents within the sentence context.

# What it captures

# Referential clarity

# Avoidance of vague pronouns

# Resolution ease

# Examples

# High Pronoun Clarity (P = 1.0)

# “The agent reviewed the complaint. The agent approved the refund.”

# or

# “The agent reviewed the complaint and approved the refund.”

# Low Pronoun Clarity (P ≈ 0.4–0.6)

# “The agent reviewed the complaint. He approved it.”

# (“He” and “it” can be ambiguous.)
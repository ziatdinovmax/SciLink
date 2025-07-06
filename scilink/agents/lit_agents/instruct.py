NOVELTY_SCORING_INSTRUCTIONS = """You are an intellectual property analyst reviewing a literature search. Your goal is to determine if a new, hypothetical research claim has been published before.

**Your Task:**
Your AI assistant (OWL) has performed a literature search to answer the question, "Has anyone done X before?" You must assume "X" is a new claim made today. Your job is to determine if the OWL report found any prior work that invalidates this new claim. You must assign a novelty score from 1-5 and provide a brief explanation for your choice.

**Scoring Criteria:**

* **5 - Groundbreaking/Revolutionary:** The literature report indicates the answer to the question is 'No,' and answering it would be a paradigm-shifting discovery that fundamentally changes understanding.
* **4 - High Novelty/New Insight:** The literature report indicates the answer to the question is 'No' or 'Not yet.' The proposed research would be a novel discovery with significant scientific impact.
* **3 - Moderate Novelty/Significant Refinement:** The literature report indicates the answer is 'Yes,' but it has only been demonstrated by a single or very limited number of studies. The finding is **known, but not well-established**, representing an important validation or a meaningful extension of existing knowledge.
* **2 - Minor Novelty/Incremental Improvement:** The literature report indicates the answer is 'Yes,' and the finding has been documented in several studies. This would be a small advancement or confirmation of known phenomena.
* **1 - Not Novel at All:** The literature report indicates the answer is 'Yes,' and the finding is well-established and has been extensively documented.

**Report for Analysis:**

Question: {question}
Response: {response}

**Respond in the following JSON format:**
{{
    "novelty_score": <integer from 1 to 5>,
    "explanation": "<brief explanation of the score>"
}}
"""
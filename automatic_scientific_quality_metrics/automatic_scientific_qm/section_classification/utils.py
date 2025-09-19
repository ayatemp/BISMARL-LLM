"""
Constants used for section classification.
"""

SECTIONS = [
    "introduction",
    "background",
    "methodology",
    "experiments and results",
    "conclusion",
]

LABEL2SECTION = {i: section for i, section in enumerate(SECTIONS)}

SECTION_SYNONYMS = {
    "introduction": ["introduction"],
    "background": ["background", "related work", "historical review"],
    "methodology": ["methodology", "method", "algorithm", "properties"],
    "experiments and results": [
        "experiments",
        "results",
        "experiments and results",
        "experimental design",
        "empirical evaluation",
        "experiments and analysis",
        "ablation studies",
        "evaluation",
    ],
    "conclusion": [
        "conclusion",
        "conclusion & discussion",
        "discussion and conclusions",
        "conclusion and outlook",
        "further work",
        "discussions and future directions",
    ],
}

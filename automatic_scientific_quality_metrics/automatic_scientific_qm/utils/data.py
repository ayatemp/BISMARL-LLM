"""
Data Models to represent Papers, References, Reviews, Comments and Sections.
"""

import logging


from fuzzywuzzy import process
from pydantic import BaseModel
from pydantic import validator


class Affiliation(BaseModel):
    laboratory: str | dict | None = None
    institution: str | dict | None = None
    location: str | dict | None = None


class Author(BaseModel):
    name: str
    affiliation: Affiliation | None = None


class TextReview(BaseModel):
    title: str | None = None
    paper_summary: str | None = None
    main_review: str | None = None
    strength_weakness: str | None = None
    questions: str | None = None
    limitations: str | None = None
    review_summary: str | None = None


class Review(BaseModel):
    review_id: str
    review: TextReview
    score: float | None = None
    confidence: float | None = None
    novelty: float | None = None
    correctness: float | None = None
    clarity: float | None = None
    impact: float | None = None
    reproducibility: float | None = None
    ethics: str | None = None

    @validator(
        "score",
        "confidence",
        "novelty",
        "correctness",
        "clarity",
        "impact",
        "reproducibility",
    )
    @classmethod
    def check_score(cls, v: float) -> float:
        if v == None:
            return v
        if v < 0 or v > 1:
            raise ValueError("score must be between 0 and 1")
        return v


class Comment(BaseModel):
    title: str | None = None
    comment: str


class Reference(BaseModel):
    # Basic paper info
    paperhash: str
    title: str
    abstract: str = ""
    authors: list[Author]

    # IDs
    arxiv_id: str | None = ""
    s2_corpus_id: str | None = ""
    external_ids: dict | None = {}

    # Reference specific info
    intents: list[str] | None = None
    isInfluential: bool | None = None


class Section(BaseModel):
    name: str
    sec_num: str
    classification: str = ""
    text: str
    subsections: list["Section"] = []


class Paper(BaseModel):
    # Basic paper info
    title: str
    authors: list[Author]
    abstract: str | None = None
    summary: str | None = None

    # Processing info
    updated: bool = False

    # ID's
    paperhash: str
    arxiv_id: str | None = None
    s2_corpus_id: str | None = ""

    # OpenReview Metadata
    field_of_study: list[str] | str | None = None
    venue: str | None = None
    publication_date: str | None = None

    # s2 Metadata
    n_references: int | None = None
    n_citations: int | None = None
    n_influential_citations: int | None = None
    open_access: bool | None = None
    external_ids: dict | None = None
    pdf_url: str | None = None

    # Content
    parsed_pdf: dict | None = None
    parsed_latex: dict | None = None
    structured_content: dict[str, Section] = {}

    # Review Data
    openreview: bool
    decision: bool | None = None
    decision_text: str | None = None
    reviews: list[Review] | None = None
    comments: list[Comment] | None = None

    # References
    references: list[Reference] | None = None
    bibref2section: dict = {}
    bibref2paperhash: dict = {}

    # Hypothesis
    hypothesis: dict | None = None

    def organize_text(self) -> None:
        """
        Organizes the parsed_pdf into a dictionary.
        """

        conclusion_passed = False
        last_sec_num = "Unnumbered"
        if (
            "pdf_parse" not in self.parsed_pdf
            or "body_text" not in self.parsed_pdf["pdf_parse"]
        ):
            self.structured_content = {}
            return

        for part in self.parsed_pdf["pdf_parse"]["body_text"]:
            # Update conclusion passed
            if not conclusion_passed and len(self.structured_content) > 0:
                sec_names = [sec.name for sec in self.structured_content.values()]
                match, _ = fuzzy_matching("conclusion", sec_names)
                if match != "":
                    conclusion_passed = True

            sec_num = part.get("sec_num")
            section = part.get("section")
            section = section.lower() if section is not None else section

            if sec_num is None:
                sec_num = "appendix" if conclusion_passed else last_sec_num
            else:
                last_sec_num = sec_num

            main_sec, _, sub_sec = sec_num.partition(".")

            # Main sec is not present yet
            if main_sec not in self.structured_content:
                self.structured_content[main_sec] = Section(
                    name=section, sec_num=main_sec, text=part["text"]
                )
                if sub_sec != None:
                    self.structured_content[main_sec].subsections.append(
                        Section(name=section, sec_num=sub_sec, text=part["text"])
                    )
            else:
                # Add text to the section
                self.structured_content[main_sec].text += part["text"]
                if sub_sec != None:
                    # Add subsection
                    self.structured_content[main_sec].subsections.append(
                        Section(name=section, sec_num=sub_sec, text=part["text"])
                    )

    def get_text(self, with_appendix: True) -> str:
        """
        Return the text of the paper.
        """
        text = ""

        paper_no_appendix = {
            key: value
            for key, value in self.structured_content.items()
            if value.sec_num != "appendix"
        }
        paper_appendix = {
            key: value
            for key, value in self.structured_content.items()
            if value.sec_num == "appendix"
        }

        for value in sorted(
            paper_no_appendix.values(),
            key=lambda y: int(y.sec_num) if y.sec_num != "Unnumbered" else float("inf"),
        ):
            text += f"{value.name}: \n {value.text} \n"

        appendix = ""
        for value in paper_appendix.values():
            appendix += value.text

        if with_appendix:
            text = f"Main paper: \n {text} \n Appendix: {appendix} \n"
        else:
            text = f"Main paper: \n {text} \n"

        return text

    def get_section_names(self) -> list[str]:
        section_names = []
        for section in self.structured_content.values():
            section_names.append(section.name)
        return section_names

    def get_section_by_name(self, section_name: str) -> str:
        for section in self.structured_content.values():
            if section.name == section_name:
                return section.text

        return ""

    def get_section_by_classification(self, section_type: str) -> str:
        for section in self.structured_content.values():
            if section.classification == section_type:
                return section.text
        return ""


def fuzzy_matching(
    query: str, choices: list[str], threshold: int = 80
) -> tuple[str, int]:
    """
    Fuzzy matches a query against a list of choices.

    Args:
        query (str): The query to match.
        choices (list[str]): A list of choices to match against.

    Returns:
        str: The best match from the list of choices.
    """

    result = process.extractOne(query, choices)

    if result is None:
        return "", 0
    else:
        best_match, score = result

    if score < threshold:
        logging.info("No match found that exceeds the threshold")
        return "", score

    return best_match, score

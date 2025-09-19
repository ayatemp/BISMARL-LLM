"""
Run the SAKANA reviewer for a subsample of the NEURIPS and ICLR 2024 papers
"""

import argparse
import logging
import json
import os

import tqdm
import yaml

from typing import Union

import anthropic
import numpy as np
import json
import openai

from datasets import load_from_disk

from automatic_scientific_qm.utils.data import Paper
from automatic_scientific_qm.llm_reviewing.sakana_utilities import (
    get_batch_responses_from_llm,
    get_response_from_llm,
    extract_json_between_markers,
)
from automatic_scientific_qm.llm_reviewing.prompts import (
    NEURIPS_FORM,
    ICLR_FORM,
    FEW_SHOT_PAPERS_ICLR,
    FEW_SHOT_REVIEWS_ICLR,
    FEW_SHOT_PAPERS_NEURIPS,
    FEW_SHOT_REVIEWS_NEURIPS,
    NEURIPS_FORM,
    REVIEWER_SYSTEM_PROMPT_BASE,
    META_REVIEWER_SYSTEM_PROMPT,
    REVIEWER_REFLECTION_PROMPT,
)

"""
Part of the Code in this file is adapted from the AI-Scientist project by SakanaAI (https://github.com/SakanaAI/AI-Scientist), licensed under the Apache License, Version 2.0.

Copyright 2024 SakanaAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def run_evaluation(config: dict) -> None:
    """
    Run the evaluation process based on the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the evaluation.

    Raises:
        ValueError: If the dataset specified in the configuration is not recognized.

    Returns:
        None
    """
    # Load dataset
    if config["dataset"] == "iclr":
        dataset = load_from_disk("../../data/processed/openreview-iclr")
        paperhash2data = {sample["paperhash"]: sample for sample in dataset["test"]}

    elif config["dataset"] == "neurips":
        dataset = load_from_disk("../../data/processed/openreview-neurips")
        paperhash2data = {sample["paperhash"]: sample for sample in dataset["test"]}

    else:
        raise ValueError(f"Dataset {config['dataset']} not recognized")

    if config["dataset"] == "iclr":
        form = ICLR_FORM
    elif config["dataset"] == "neurips":
        form = NEURIPS_FORM
    else:
        raise ValueError(f"Dataset {config['dataset']} not recognized")

    client = get_client(config)

    with open(config["venue_file"], "r") as f:
        paperhashes = json.load(f)

    # If part of the dataset has already been reviewed, load the existing reviews
    if os.path.exists(config["output_file"]):
        with open(config["output_file"], "r") as f:
            paperhash2review = json.load(f)
    else:
        paperhash2review = {}

    for paperhash in tqdm.tqdm(paperhashes):
        if paperhash in paperhash2review:
            continue
        try:
            paper_sample = paperhash2data[paperhash]

        except Exception as e:
            logging.info(f"Error loading paper {paperhash}: {e}")
            continue

        text = paper_sample["full_text"]

        review = perform_review(
            text,
            config["model"],
            client,
            config["num_reflections"],
            config["num_fs_examples"],
            config["num_reviews_ensemble"],
            config["temperature"],
            review_instruction_form=form,
        )

        paperhash2review[paperhash] = review

        with open(config["output_file"], "w") as f:
            json.dump(paperhash2review, f, indent=4)


def perform_review(
    text: str,
    model: str,
    client: openai.Client,
    num_reflections: int = 1,
    num_fs_examples: int = 2,
    num_reviews_ensemble: int = 1,
    temperature: float = 0.75,
    msg_history: Union[list, None] = None,
    return_msg_history: bool = False,
    reviewer_system_prompt: str = REVIEWER_SYSTEM_PROMPT_BASE,
    review_instruction_form: str = NEURIPS_FORM,
) -> Union[dict, tuple[dict, list]]:
    """
    Perform a review of a given text using the LLM model.

    Args:
        text (str): The text to be reviewed.
        model (str): The LLM model to use for the review.
        client (openai.Client): The OpenAI client for making API requests.
        num_reflections (int, optional): The number of reviewer reflections to perform. Defaults to 1.
        num_fs_examples (int, optional): The number of few-shot examples to include in the review. Defaults to 2.
        num_reviews_ensemble (int, optional): The number of reviews to generate and aggregate. Defaults to 1.
        temperature (float, optional): The temperature parameter for controlling the randomness of the model's output. Defaults to 0.75.
        msg_history (list or None, optional): The message history for the conversation with the model. Defaults to None.
        return_msg_history (bool, optional): Whether to return the message history along with the review. Defaults to False.
        reviewer_system_prompt (str, optional): The system prompt for the reviewer. Defaults to REVIEWER_SYSTEM_PROMPT_BASE.
        review_instruction_form (str, optional): The form for providing review instructions. Defaults to NEURIPS_FORM.

    Returns:
        Union[dict, tuple[dict, list]]: The generated review as a dictionary or a tuple containing the review and the message history.
    """

    if num_fs_examples > 0:
        fs_prompt = get_review_fewshot_examples(num_fs_examples, iclr=True)
        base_prompt = review_instruction_form + fs_prompt
    else:
        base_prompt = review_instruction_form

    base_prompt += f"""
    Here is the paper you are asked to review:
    ```
    {text}
    ```"""

    if num_reviews_ensemble > 1:
        llm_review, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=0.75,
            n_responses=num_reviews_ensemble,
        )
        parsed_reviews = []
        for idx, rev in enumerate(llm_review):
            try:
                parsed_reviews.append(extract_json_between_markers(rev))
            except Exception as e:
                print(f"Ensemble review {idx} failed: {e}")
        parsed_reviews = [r for r in parsed_reviews if r is not None]
        review = get_meta_review(
            model, client, temperature, parsed_reviews, review_instruction_form
        )

        # take first valid in case meta-reviewer fails
        if review is None:
            review = parsed_reviews[0]

        # Replace numerical scores with the average of the ensemble.
        for score, limits in [
            ("Originality", (1, 4)),
            ("Quality", (1, 4)),
            ("Clarity", (1, 4)),
            ("Significance", (1, 4)),
            ("Soundness", (1, 4)),
            ("Presentation", (1, 4)),
            ("Contribution", (1, 4)),
            ("Overall", (1, 10)),
            ("Confidence", (1, 5)),
        ]:
            scores = []
            for r in parsed_reviews:
                if score in r and limits[1] >= r[score] >= limits[0]:
                    scores.append(r[score])
            review[score] = int(round(np.mean(scores)))

        # Rewrite the message history with the valid one and new aggregated review.
        msg_history = msg_histories[0][:-1]
        msg_history += [
            {
                "role": "assistant",
                "content": f"""
                            THOUGHT:
                            I will start by aggregating the opinions of {num_reviews_ensemble} reviewers that I previously obtained.

                            REVIEW JSON:
                            ```json
                            {json.dumps(review)}
                            ```
                            """,
            }
        ]
    else:
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = extract_json_between_markers(llm_review)

    if num_reflections > 1:
        for j in range(num_reflections - 1):
            text, msg_history = get_response_from_llm(
                REVIEWER_REFLECTION_PROMPT,
                client=client,
                model=model,
                system_message=reviewer_system_prompt,
                msg_history=msg_history,
                temperature=temperature,
            )
            review = extract_json_between_markers(text)
            assert review is not None, "Failed to extract JSON from LLM output"

            if "I am done" in text:
                break

    if return_msg_history:
        return review, msg_history
    else:
        return review


def get_review_fewshot_examples(num_fs_examples=1, iclr=False) -> str:
    """
    Generates a few-shot prompt with sample reviews from previous machine learning conferences.

    Args:
        num_fs_examples (int): Number of few-shot examples to include in the prompt. Default is 1.
        iclr (bool): If True, uses few-shot papers and reviews from ICLR conference.
                     If False, uses few-shot papers and reviews from NeurIPS conference. Default is False.

    Returns:
        str: Few-shot prompt with sample reviews.

    """
    fewshot_prompt = """
    Below are some sample reviews, copied from previous machine learning conferences.
    Note that while each review is formatted differently according to each reviewer's style, the reviews are well-structured and therefore easy to navigate.
    """

    if iclr:
        few_shot_papers = FEW_SHOT_PAPERS_ICLR
        few_shot_reviews = FEW_SHOT_REVIEWS_ICLR
    else:
        few_shot_papers = FEW_SHOT_PAPERS_NEURIPS
        few_shot_reviews = FEW_SHOT_REVIEWS_NEURIPS

    for paper_text, review_text in zip(
        few_shot_papers[:num_fs_examples], few_shot_reviews[:num_fs_examples]
    ):
        fewshot_prompt += f"""
        Paper:

        ```
        {paper_text}
        ```

        Review:

        ```
        {review_text}
        ```

        """

    return fewshot_prompt


def get_meta_review(
    model: str,
    client: openai.Client,
    temperature: float,
    reviews: list,
    review_form: str,
) -> dict:
    """
    Generate a meta-review from a set of individual reviews.

    Args:
        model (str): The model to use for generating the meta-review.
        client (openai.Client): The client to use for generating the meta-review.
        temperature (float): The temperature parameter for controlling the randomness of the generated text.
        reviews (list): A list of individual reviews.
        review_form (str): The review form to use as the base prompt for generating the meta-review.

    Returns:
        dict: The generated meta-review.
    """
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"""
        Review {i + 1}/{len(reviews)}:
        ```
        {json.dumps(r)}
        ```
        """
    base_prompt = review_form + review_text

    llm_review, _ = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=META_REVIEWER_SYSTEM_PROMPT.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    meta_review = extract_json_between_markers(llm_review)
    return meta_review


def get_client(config):
    if config["llm_provider"] == "openai":
        assert (
            "OPENAI_API_KEY" in os.environ
        ), "Please set the OPENAI_API_KEY environment variable"
        config["model"] = config["openai_model"]
        return openai.Client(api_key=os.environ["OPENAI_API_KEY"])

    elif config["llm_provider"] == "anthropic":
        assert (
            "ANTHROPIC_API_KEY" in os.environ
        ), "Please set the ANTHROPIC_API_KEY environment variable"
        config["model"] = config["anthropic_model"]
        return anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--llm_provider", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    if args.llm_provider is not None:
        config["llm_provider"] = args.llm_provider

    # Setup logging
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        filename="logs/sakana.log",
        filemode="a+",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    run_evaluation(config)

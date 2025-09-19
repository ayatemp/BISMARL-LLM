"""
Utilities to run a swiss torunament
"""

import json
import os
import random

import numpy as np
import pandas as pd
import retry
import torch
import wandb

from collections import defaultdict
from typing import Union

from anthropic import Anthropic
from openai import OpenAI
from tqdm import tqdm

from automatic_scientific_qm.score_prediction.data import transform_data_factory
from automatic_scientific_qm.score_prediction.trainer import PairwiseComparison


"""
This code includes portions from https://github.com/NoviScl/AI-Researcher by Chenglei Si,
licensed under the MIT License (see below or at https://github.com/NoviScl/AI-Researcher?tab=MIT-1-ov-file#readme).
Copyright (c) 2024 Chenglei Si
"""


def call_api(
    client: Union[Anthropic, OpenAI],
    model: str,
    prompt_messages: str,
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    seed=2024,
    json_output=False,
) -> tuple[str, int]:
    """
    Calls the API to generate a response using the specified client and model.

    Args:
        client (Union[Anthropic, OpenAI]): The client object used to interact with the LLM API.
        model (str): The name of the LLM model.
        prompt_messages (str): The prompt messages to use.
        temperature (float, optional): The temperature parameter for controlling the randomness of the output. Defaults to 1.0.
        top_p (float, optional): The top-p parameter for controlling the diversity of the output. Defaults to 1.0.
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 1000.
        seed (int, optional): The seed value for generating random numbers. Defaults to 2024.
        json_output (bool, optional): Whether to output the response as a JSON object. Defaults to False.

    Returns:
        tuple[str,int]: A tuple containing the generated response and an error code (always 0 in this case).
    """

    ## Anthropic models
    if "claude" in model:
        if json_output:
            prompt = (
                prompt_messages[0]["content"]
                + ' Directly output the JSON dict with no additional text (avoid the presence of newline characters ("\n") and unescaped double quotes within the string so that we can call json.loads() on the output directly). Make sure you follow the exact same JSON format as shown in the examples. Don\'t include "```json" or "```" at the beginning and end of the output.'
            )
            prompt_messages = [{"role": "user", "content": prompt}]
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=prompt_messages,
        )

        response = message.content[0].text

    ## OpenAI models
    else:
        ## o1 models
        if "o1" in model:
            if json_output:
                prompt = (
                    prompt_messages[0]["content"]
                    + ' Directly output the JSON dict with no additional text (avoid the presence of newline characters ("\n") and unescaped double quotes within the string so that we can call json.loads() on the output directly). Make sure you follow the exact same JSON format as shown in the examples. Don\'t include "```json" or "```" at the beginning and end of the output.'
                )
                prompt_messages = [{"role": "user", "content": prompt}]
            completion = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                max_completion_tokens=max_tokens,
                seed=seed,
            )
            # print ("completion: ", completion)
        ## 4o and other OpenAI models
        else:
            response_format = (
                {"type": "json_object"} if json_output else {"type": "text"}
            )
            completion = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                response_format=response_format,
            )

        response = completion.choices[0].message.content.strip()

    return response, 0


@retry.retry(tries=3, delay=2)
def better_idea_llm(
    paper1: dict,
    paper2: dict,
    client: Union[Anthropic, OpenAI],
    model: str,
    seed: int,
    conference: str,
    temperature: float = 0.0,
):
    """
    Determines which of the two submitted papers is better and should receive the higher overall score.

    Args:
        paper1 (Paper): The first submitted paper.
        paper2 (Paper): The second submitted paper.
        client: LLM API.
        model (str): LLM model name.
        seed (int): The seed value for random number generation.
        conference (str): The name of the conference.
        temperature (float, optional): The temperature value for controlling the randomness of the generated responses. Defaults to 0.0.

    Returns:
        tuple: A tuple containing the prompt, response, and cost.
            - prompt (str): The prompt message used for generating the response.
            - response (str): The generated response.
            - cost (float): The cost of the API call.
    """
    paper1_text = paper1["full_text"]
    paper2_text = paper2["full_text"]

    prompt = f"You are a reviewer for the conference {conference} You are given two submitted papers. Decide which of them is better and should receive the higher overall score.\n"

    prompt += "The two project proposals are:\n\n"
    prompt += "paper 1:\n" + paper1_text + "\n\n"
    prompt += "paper 2:\n" + paper2_text + "\n\n"
    prompt += "Now decide which one is the better paper. Directly return a number 1 or 2 and nothing else.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(
        client,
        model,
        prompt_messages,
        temperature=temperature,
        max_tokens=3000,
        seed=seed,
        json_output=False,
    )
    return prompt, response, cost


def better_idea_rsp(
    paper1: dict,
    paper2: dict,
    rsp_model: PairwiseComparison,
    paperhash2sample: dict,
    config: dict,
) -> str:
    """
    Determines the better idea between two papers using a pairwise comparison model.

    Args:
        paper1 (Paper): The first paper to compare.
        paper2 (Paper): The second paper to compare.
        rsp_model (PairwiseComparison): The pairwise comparison model.
        paperhash2sample (dict): A dictionary mapping paperhashes to samples used as model input.
        config (dict): Configuration parameters.

    Returns:
        str: The label of the better paper according to the model. Returns "1" if paper1 is better, and "2" if paper2 is better.
    """
    paper_representation = f"{config['paper_representation']}_emb"
    paper_representations = [
        torch.tensor(paper1[paper_representation]).float(),
        torch.tensor(paper2[paper_representation]).float(),
    ]
    references = [
        torch.tensor(paper1["reference_emb"]).float(),
        torch.tensor(paper2["reference_emb"]).float(),
    ]
    full_paper1 = np.stack(
        [
            paper1["intro_emb"],
            paper1["method_emb"],
            paper1["result_experiment_emb"],
            paper1["background_emb"],
            paper1["conclusion_emb"],
        ],
        axis=0,
    )
    full_paper1 = torch.tensor(full_paper1).float()
    full_paper2 = np.stack(
        [
            paper2["intro_emb"],
            paper2["method_emb"],
            paper2["result_experiment_emb"],
            paper2["background_emb"],
            paper2["conclusion_emb"],
        ],
        axis=0,
    )
    full_paper2 = torch.tensor(full_paper2).float()
    full_papers = [full_paper1, full_paper2]
    scores = torch.tensor([paper1["mean_score"], paper2["mean_score"]]).float()

    transform_config = {
        "data": {"pairwise_comparison": True, "context_type": rsp_model.context_type}
    }
    transform_data = transform_data_factory(transform_config)
    transformed_data = transform_data(
        paper_representations, references, full_papers, scores
    )
    transformed_data = [t.to(config["device"]) for t in transformed_data]

    with torch.no_grad():
        predicted_comparison = rsp_model.predict(transformed_data)

        if predicted_comparison == 0:
            return "2"
        else:
            return "1"


def single_round(
    papers: list[dict],
    scores: dict,
    client: Union[Anthropic, OpenAI, None],
    model: Union[PairwiseComparison, str],
    seed: int,
    conference: str,
    paperhash2sample: dict,
    current_round: int,
    all_costs: float,
    config: dict,
) -> tuple[float, int, int]:
    """
    Runs a single round of the Swiss tournament algorithm.

    Args:
        papers (list[Paper]): A list of Paper objects representing the papers to be compared.
        scores (dict): A dictionary mapping paperhashes to their current scores.
        client (Union[Anthropic, OpenAI]): The client object used for making LLM API calls.
        model (Union[PairwiseComparison, str]): The model used for pairwise comparison.
        seed (int): The seed value for random number generation.
        conference (str): The name of the conference.
        paperhash2sample (dict): A dictionary mapping paperhashes to their corresponding samples.
        current_round (int): The current round number.
        all_costs (float): The total cost incurred so far.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        tuple[float, int, int]: A tuple containing the total cost, number of comparisons, and number of correct comparisons.
    """

    # Shuffle ideas in the first round
    if current_round == 0:
        random.shuffle(papers)

    match_pairs = []

    # Sort ideas based on current scores
    sorted_papers = sorted(
        papers, key=lambda paper: scores[paper["paperhash"]], reverse=True
    )

    for i in range(0, len(sorted_papers), 2):
        if i + 1 < len(sorted_papers):
            match_pairs.append((sorted_papers[i], sorted_papers[i + 1]))
        else:
            # If there is an odd number of ideas, the last one automatically wins this round
            scores[papers[i].paperhash] += 1

    n_comparisons = 0
    correct_comparisons = 0
    for paper1, paper2 in tqdm(match_pairs):
        if config["model_type"] == "llm":
            _, result, cost = better_idea_llm(
                paper1, paper2, client, model, seed, conference
            )
        elif config["model_type"] == "rsp":
            result = better_idea_rsp(paper1, paper2, model, paperhash2sample, config)
            cost = 0
        else:
            raise NotImplementedError("Only llm and rsp are supported")

        paper1_review_score = paper1["mean_score"]
        paper2_review_score = paper2["mean_score"]

        if result.strip() == "1":
            scores[paper1["paperhash"]] += 1

            if paper1_review_score >= paper2_review_score:
                correct_comparisons += 1

        else:
            scores[paper2["paperhash"]] += 1

            if paper2_review_score >= paper1_review_score:
                correct_comparisons += 1

        n_comparisons += 1
        all_costs += cost

    return all_costs, n_comparisons, correct_comparisons


def tournament_ranking(
    papers: list[dict],
    model: Union[PairwiseComparison, str],
    seed: int,
    conference: str,
    paperhash2sample: dict,
    max_round: int,
    config: dict,
) -> None:
    """
    Runs a Swiss tournament ranking for a given set of papers.

    Args:
        papers (list[Paper]): List of Paper objects representing the papers in the tournament.
        model (Union[PairwiseComparison,str]): The model used for pairwise comparisons.
        seed (int): The seed value for random number generation.
        conference (str): The name of the conference.
        paperhash2sample (dict): A dictionary mapping paper hashes to samples.
        max_round (int): The maximum number of rounds to run the tournament.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        float: The total cost of the tournament.

    Raises:
        None
    """
    # Create result directory
    os.makedirs("./results", exist_ok=True)

    # Load client
    if config["model_type"] == "llm":
        if config["llm_provider"] == "anthropic":
            assert "ANTHROPIC_API_KEY" in os.environ, "No ANTHROPIC_API_KEY found"
            key = os.environ["ANTHROPIC_API_KEY"]
            client = Anthropic(
                api_key=key,
            )
        elif config["llm_provider"] == "openai":
            assert "OPENAI_API_KEY" in os.environ, "No OPEN_API_KEY found"
            key = os.environ["OPENAI_API_KEY"]
            client = OpenAI(api_key=key)
        else:
            raise NotImplementedError("Only claude and openai are supported")
    else:
        client = None

    total_comparisons = 0
    total_correct_comparisons = 0
    scores = defaultdict(int)
    all_costs = 0
    current_round = 0

    # Run swiss tournament
    while current_round < max_round:
        print("Current round: ", current_round + 1)
        all_costs, n_comparisons, correct_comparisons = single_round(
            papers,
            scores,
            client,
            model,
            seed,
            conference,
            paperhash2sample,
            current_round=current_round,
            all_costs=all_costs,
            config=config,
        )
        total_comparisons += n_comparisons
        total_correct_comparisons += correct_comparisons

        with open(
            f"./results/scores_{config['dataset']}_{current_round}_{config['model_type']}_{config['llm_provider']}.json",
            "w",
        ) as file:
            json.dump(scores, file, indent=4)

        current_round += 1

    for paper in papers:
        scores[paper["paperhash"]] = (scores[paper["paperhash"]], paper["mean_score"])

    # Log final scores
    with open(
        f"./results/final_scores_{config['dataset']}_{current_round}_{config['model_type']}_{config['llm_provider']}.json",
        "w",
    ) as file:
        json.dump(scores, file, indent=4)

    with open(
        f"./results/comparisons_{config['dataset']}_{current_round}_{config['model_type']}_{config['llm_provider']}.json",
        "w",
    ) as file:
        json.dump(
            {
                "comparisons": total_comparisons,
                "correct_comparisons": total_correct_comparisons,
            },
            file,
            indent=4,
        )

    # Compute spearman correlation between comparison scores and review scores
    comparison_scores = []
    review_scores = []
    for score in scores.values():
        comparison_scores.append(score[0])
        review_scores.append(score[1])

    df = pd.DataFrame(
        {"comparison_scores": comparison_scores, "review_scores": review_scores}
    )
    wandb.log({"spearman_correlation": df.corr(method="spearman").iloc[0, 1]})

    # Comparison Accuracy
    wandb.log({"comparison_accuracy": total_correct_comparisons / total_comparisons})

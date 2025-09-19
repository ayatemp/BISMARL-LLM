"""
Download the OpenReview datasets from HuggingFace and add the parsed and annotated pdfs to the dataset, save locally.
"""

import argparse
import json
import logging
import os

import tqdm
import yaml

from datasets import load_dataset, Dataset, DatasetDict
from urllib.error import HTTPError
from urllib.request import urlretrieve
from doc2json.grobid2json.grobid.grobid_client import (
    GrobidClient,
)
from doc2json.grobid2json.tei_to_json import (
    convert_tei_xml_file_to_s2orc_json,
)

from automatic_scientific_qm.section_classification.trainer import SectionClassifier
from automatic_scientific_qm.data_processing.utils import (
    parsedpdf2structured_content,
    annotate,
    StructuredContent,
    FEATURES,
)


def complete_openreview_dataset(config: dict) -> None:
    # Create save directory
    save_directory = os.path.join(config["save_directory"], config["dataset_name"])
    os.makedirs(save_directory, exist_ok=True)

    # Load existing
    if os.path.exists("./data/paperhash2structured_content.json"):
        with open("./data/paperhash2structured_content.json", "r") as f:
            paperhash2structured_content = json.load(f)
            for key, value in paperhash2structured_content.items():
                paperhash2structured_content[key] = StructuredContent.load_json(value)
    else:
        paperhash2structured_content = {}

    n_new_samples = 0

    # Setup GROBID client for pdf parsing
    grobid_client = GrobidClient()
    grobid_client.max_workers = config["max_workers"]

    # Prepare PDF directory
    pdf_directory = os.path.join(config["save_directory"], "pdfs")
    os.makedirs(pdf_directory, exist_ok=True)

    # Prepare XMLs directory
    xml_directory = os.path.join(config["save_directory"], "xmls")
    os.makedirs(xml_directory, exist_ok=True)

    # Prepare data directory
    os.makedirs("data", exist_ok=True)

    # Load data and model
    dataset = load_dataset(
        "nhop/scientific-quality-score-prediction", config["dataset_name"]
    )
    section_classifier = SectionClassifier.load_from_checkpoint(
        "./model_store/section_classifier_openreview.ckpt",
        map_location=config["device"],
    )
    section_classifier.load_preprocessing_utils(config["device"])
    section_classifier.eval()

    # Iterate over samples in the dataset
    new_dataset_dict = {}
    for split in ["train", "validation", "test"]:
        new_samples = []
        for sample in tqdm.tqdm(dataset[split]):
            # If we cannot extract segments set them to None
            introduction = None
            background = None
            methodology = None
            experiments_results = None
            conclusion = None
            full_text = None

            # Check wheter pdf has already been parsed
            if sample["paperhash"] in paperhash2structured_content:
                structured_content = paperhash2structured_content[sample["paperhash"]]
            else:
                # Obtain the pdf
                pdf_id = sample["openreview_submission_id"]

                try:
                    pdf_filename = os.path.join(
                        pdf_directory, f"{sample['paperhash']}.pdf"
                    )
                    if not os.path.exists(pdf_filename):
                        url = f"https://openreview.net/pdf?id={pdf_id}"
                        urlretrieve(url, pdf_filename)

                except Exception as e:
                    logging.warning(
                        f"Could not download the pdf for submission {sample['paperhash'] }."
                    )
                    logging.warning(e)
                    pdf_filename = None

                if pdf_filename != None:
                    try:
                        # PDF -> XML
                        grobid_client.process_batch(
                            [pdf_filename], xml_directory, "processFulltextDocument"
                        )

                        # XML -> JSON
                        xml_filename = os.path.join(
                            xml_directory, f"{sample['paperhash']}.tei.xml"
                        )
                        parsed_pdf = convert_tei_xml_file_to_s2orc_json(xml_filename)
                        parsed_pdf = parsed_pdf.release_json()

                        # JSON -> structured content
                        structured_content = parsedpdf2structured_content(parsed_pdf)
                        structured_content = annotate(
                            structured_content, section_classifier, config
                        )
                        n_new_samples += 1
                    except Exception as e:
                        logging.warning(
                            f"Could not parse the pdf for submission {sample['paperhash']}."
                        )
                        logging.warning(e)
                        structured_content = StructuredContent({})
                        n_new_samples += 1

            # Add to paperhash2structured_content
            paperhash2structured_content[sample["paperhash"]] = structured_content

            # print(structured_content.structured_content)
            if structured_content.structured_content is not None:
                # Annotate the pdf with the section classifier
                introduction = structured_content.get_section_by_classification(
                    "introduction"
                )
                background = structured_content.get_section_by_classification(
                    "background"
                )
                methodology = structured_content.get_section_by_classification(
                    "methodology"
                )
                experiments_results = structured_content.get_section_by_classification(
                    "experiments_results"
                )
                conclusion = structured_content.get_section_by_classification(
                    "conclusion"
                )
                full_text = structured_content.get_full_text()

                if len(structured_content.structured_content) > 0:
                    logging.info(f"Parsed and annotated the pdf {sample['paperhash']}.")

            sample["introduction"] = introduction
            sample["background"] = background
            sample["methodology"] = methodology
            sample["experiments_results"] = experiments_results
            sample["conclusion"] = conclusion
            sample["full_text"] = full_text
            new_samples.append(sample)

            # Save the paperhash2structured_content
            if n_new_samples % 1000 == 1:
                with open("./data/paperhash2structured_content.json", "w") as f:
                    paperhash2structured_content_json = {
                        key: value.to_json()
                        for key, value in paperhash2structured_content.items()
                    }
                    json.dump(paperhash2structured_content_json, f, indent=4)

        new_dataset_dict[split] = Dataset.from_list(new_samples, features=FEATURES)

    # Save the latest paperhash2structured_content
    with open("./data/paperhash2structured_content.json", "w") as f:
        paperhash2structured_content_json = {
            key: value.to_json() for key, value in paperhash2structured_content.items()
        }
        json.dump(paperhash2structured_content_json, f, indent=4)

    # Save the new_dataset
    dataset = DatasetDict(new_dataset_dict)
    dataset.save_to_disk(save_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.dataset is not None:
        config["dataset_name"] = args.dataset

    # Create logs directory if needed
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler("logs/complete_dataset.log")],
    )

    complete_openreview_dataset(config)

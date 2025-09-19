"""
Data loading utilities for section classification.
"""

import os

import nltk
import torch

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from adapters import AutoAdapterModel


class SectionClassifierDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample["embedding"], sample["label"]

    def __len__(self):
        return len(self.dataset)


def collate_section_classifier(batch):
    embeddings = []
    labels = []
    lengths = []
    for elem in batch:
        embedding, label = elem
        embedding = torch.tensor(embedding, dtype=torch.float)
        embeddings.append(embedding)
        lengths.append(embedding.shape[0])
        labels.append(label)

    embeddings = pad_sequence(embeddings, batch_first=True)
    B, T, _ = embeddings.shape
    mask = ~(
        torch.arange(T).expand(B, T)
        < torch.tensor(lengths, dtype=torch.long).unsqueeze(1)
    )
    labels = torch.tensor(labels, dtype=torch.long)
    batch = {"embeddings": embeddings, "labels": labels, "mask": mask}

    return batch


def get_data(config):
    need_compute_embeddings = not os.path.exists(
        os.path.join(config["data"]["output_directory"], "dataset_dict.json")
    )

    if need_compute_embeddings:
        dataset = load_dataset("nhop/academic-section-classification")

        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoAdapterModel.from_pretrained("allenai/specter2_base").to(
            config["device"]
        )
        model.load_adapter(
            "allenai/specter2", source="hf", load_as="classification", set_active=True
        )
        model = model.to(config["device"])

        def compute_embeddings(sample):
            sentences = nltk.tokenize.sent_tokenize(sample["text"])
            sentences = sentences[: config["data"]["max_n_sentences_per_example"]]
            inputs = tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(config["device"])

            with torch.no_grad():
                outputs = model(**inputs)

            return {"embedding": outputs.last_hidden_state.mean(dim=1).cpu().numpy()}

        dataset = dataset.map(compute_embeddings, batched=False)
        dataset.save_to_disk(config["data"]["output_directory"])

    else:
        dataset = load_from_disk(config["data"]["output_directory"])

    # Create dataloaders
    train_dataset = SectionClassifierDataset(dataset["train"])
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_section_classifier,
    )

    val_dataset = SectionClassifierDataset(dataset["validation"])
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_section_classifier,
    )

    test_dataset = SectionClassifierDataset(dataset["test"])
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_section_classifier,
    )

    return train_dataloader, val_dataloader, test_dataloader

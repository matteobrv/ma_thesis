import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel # type: ignore
from sentence_transformers import SentenceTransformer
from transformers.tokenization_utils_base import BatchEncoding
from typing import Any, List, Dict

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    """Custom dataset class to load the quadruples tense and agreement datasets.

    Args:
        df (pandas.DataFrame): The input pandas DataFrame containing the dataset.
        lm (str): The name of the Pretrained Language Model (PLM).
        lm_type (str): The type of PLM (BERT, RoBERTa, or Sentence Transformers).
        hidden_state (int): The index of the hidden state to extract the embedding from
            (e.g., -1 for last_hidden_state).
    """
    def __init__(
        self, df: pd.DataFrame, lm: str, lm_type: str, hidden_state: int
    ) -> None:
        self._lm_type = lm_type
        self._tokenizer = AutoTokenizer.from_pretrained(lm)
        if self._lm_type in ["bert", "roberta"]:
            self._model = AutoModel.from_pretrained(lm).to(DEVICE)
        elif self._lm_type == "sentence-transformers":
            self._model = SentenceTransformer(lm).to(DEVICE)
        else:
            raise ValueError("unsupported model.")
        self._df = df
        # note: -1 returns the last_hidden_state, -2 the second-to-last...
        self._hidden_state = hidden_state

    def _tokenize_sample(self, sample: List[str]) -> BatchEncoding:
        """Tokenizes the provided sample of sentences."""
        tokenized = self._tokenizer(
            sample,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        return tokenized

    def __len__(self) -> int:
        return len(self._df)

    @torch.no_grad()
    def _get_embedding(self, tokenized: BatchEncoding, sample: List[str]) -> torch.Tensor:
        """Extracts the embeddings for the given sample and hidden state."""
        if self._lm_type in ["bert", "roberta"]:
            # CLS embedding for the given hidden state
            embedding = self._model(
                **tokenized, output_hidden_states=True
            )[2][self._hidden_state][:, 0, :]

        elif self._lm_type == "sentence-transformers":
            # no need for hidden state as SentTransf. has only one
            embedding = torch.Tensor(
                self._model.encode(sample, convert_to_tensor=True)
            ).to(DEVICE)

        else:
            raise ValueError(f"Unsupported model type {self._lm_type}.")

        return embedding
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Returns the sample sentences and their respective embeddings."""
        # retrieve i-th sample and tokenize its sentences
        sample = self._df.iloc[index, 1:5].to_list()
        tokenized = self._tokenize_sample(sample)
        # retrive embeddings for the 4 sentences: x, y, d1, d2
        embeddings = self._get_embedding(tokenized, sample)
        x, y, d1, d2 = embeddings

        return {"sample": sample, "x": x, "y": y, "d1": d1, "d2": d2}

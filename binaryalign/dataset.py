import torch
import itertools
from typing import Optional
from transformers import PreTrainedTokenizer
from typing import List
from tqdm import tqdm
from binaryalign.utils import parse_single_alignment


class SpanTokenDataset:
    r"""
    Initialize SpanTokenDataset.
    
    This function is adapted from the AccAlign repository:
    https://github.com/sufenlp/AccAlign/blob/main/aligner/sent_aligner.py

    Args:
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer to use.
        lines_src (`List[str]`):
            A list of strings containing the source sentences.
        lines_tgt (`List[str]`):
            A list of strings containing the target sentences.
        gold_lines (`List[str]`, *optional*):
            A list of strings containing the gold alignments.
        gold_one_index (`bool`, *optional*):
            A boolean indicating whether the gold list indexing starts at 1. 
        ignore_possible_alignments (`bool`, defaults to ):
            A boolean indicating whether we ignore the possible alignments or not. Only recommended for training purposes.
        context_sep (`str`, defaults to ` [WORD_SEP] `):
            A token used to surround the word to get the alignment of.
        do_inference (`bool`, defaults to `False`):
            A boolean indicating whether this dataset is used for inference or training.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        lines_src: List[str],
        lines_tgt: List[str],
        gold_lines: Optional[List[str]] = None,
        gold_one_index: Optional[bool] = False,
        ignore_possible_alignments: Optional[bool] = False,
        context_sep: Optional[str] = " [WORD_SEP] ",
        do_inference: Optional[bool] = False,
    ):
        
        self.ignore_possible_alignments = ignore_possible_alignments
        self.gold_one_index = gold_one_index
        self.context_sep = context_sep
        self.tokenizer = tokenizer
        self.data, self.data_2 = [], []
        self.sure, self.possible = [], []
        self.do_inference = do_inference
        
        self.prepare(lines_src, lines_tgt, gold_lines, self.data)
        self.prepare(lines_src, lines_tgt, gold_lines, self.data_2, use_reverse=True)
        if not self.do_inference:
            self.data = self.data + self.data_2
        
        
    def prepare(self, lines_src, lines_tgt, gold_lines, data, use_reverse=False):
        # One training example per source word, for both translation directions.
        for line_number, (raw_src, raw_tgt, gold_line) in enumerate(
            tqdm(zip(lines_src, lines_tgt, gold_lines), total=len(lines_src)), start=1
        ):
            original_src = raw_src.split()
            original_tgt = raw_tgt.split()
            if not original_src or not original_tgt:
                raise ValueError(f"Empty sentence at line {line_number}")
            sent_src, sent_tgt = (
                (original_tgt, original_src) if use_reverse
                else (original_src, original_tgt)
            )

            target_tokens = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(w)) for w in sent_tgt]
            target_flat = list(itertools.chain.from_iterable(target_tokens))
            target_ids = self.tokenizer.prepare_for_model(target_flat, add_special_tokens=True)["input_ids"][1:]
            target_map = [i for i, parts in enumerate(target_tokens) for _ in parts] + [-1]
            if len(target_ids) != len(target_map):
                raise ValueError(f"Unexpected target tokenization at line {line_number}")

            links = []
            sure, possible = set(), set()
            for item in gold_line.split():
                a, b = parse_single_alignment(item, self.gold_one_index)
                if not (0 <= a < len(original_src) and 0 <= b < len(original_tgt)):
                    raise ValueError(f"Out-of-range alignment {item} on line {line_number}")
                is_sure = '-' in item
                if is_sure:
                    sure.add((a, b))
                if is_sure or not self.ignore_possible_alignments:
                    possible.add((a, b))
                    links.append((b, a) if use_reverse else (a, b))
            if not use_reverse:
                self.sure.append(sure)
                self.possible.append(possible)

            sentence_examples = []
            separator = self.context_sep.strip()
            for source_index in range(len(sent_src)):
                marked = sent_src[:source_index] + [separator, sent_src[source_index], separator] + sent_src[source_index+1:]
                source_ids = self.tokenizer.prepare_for_model(
                    list(itertools.chain.from_iterable(
                        self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(w)) for w in marked
                    )), add_special_tokens=True
                )["input_ids"]
                if len(source_ids) + len(target_ids) > 512:
                    raise ValueError(
                        f"Line {line_number} exceeds 512 tokens when concatenated "
                        f"({len(source_ids)} + {len(target_ids)}). Split or filter this pair."
                    )
                sentence_examples.append(source_ids)

            # All examples from the same sentence must have equal length in inference.
            source_width = max(map(len, sentence_examples))
            if source_width + len(target_ids) > 512:
                raise ValueError(f"Line {line_number} exceeds context length")
            ids_batch, labels_batch = [], []
            map_values = [-1] * source_width + target_map
            for source_index, source_ids in enumerate(sentence_examples):
                padded_src = source_ids + [self.tokenizer.pad_token_id] * (source_width - len(source_ids))
                ids = padded_src + target_ids
                labels = [-100] * source_width + [0] * len(target_ids)
                for a, b in links:
                    if a == source_index:
                        for k, target_word in enumerate(target_map):
                            if target_word == b:
                                labels[source_width + k] = 1
                labels[-1] = -100
                ids_batch.append(ids)
                labels_batch.append(labels)

            if self.do_inference:
                all_ids = torch.tensor(ids_batch, dtype=torch.long)
                all_labels = torch.tensor(labels_batch, dtype=torch.long)
                data.append({
                    "input_ids": all_ids,
                    "attention_mask": torch.tensor([
                        [int(j < len(sentence_examples[i])) for j in range(source_width)]
                        + [1] * len(target_ids) for i in range(len(sent_src))
                    ], dtype=torch.long),
                    "labels": all_labels,
                    "bpe2word_map": torch.tensor(map_values, dtype=torch.long),
                })
            else:
                for i, ids in enumerate(ids_batch):
                    data.append({
                        "input_ids": ids,
                        "attention_mask": [int(j < len(sentence_examples[i])) for j in range(source_width)] + [1] * len(target_ids),
                        "labels": labels_batch[i],
                    })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        if self.do_inference:
            return self.data[item], self.data_2[item]
        else:
            return self.data[item]

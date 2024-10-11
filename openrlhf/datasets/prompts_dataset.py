from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        return_raw=False,
        max_length=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        self.n_samples_per_prompt = getattr(self.strategy.args, "n_samples_per_prompt", 1)
        self.return_raw = return_raw
        self.max_length = max_length

        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.raw_prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)

            if self.max_length is not None:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                prompt = self.tokenizer.decode(prompt_token["input_ids"][0])

            self.prompts.append(prompt)
            self.raw_prompts.append(data[input_key])

    def __len__(self):
        length = len(self.prompts)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        if self.return_raw:
            return self.prompts[idx // self.n_samples_per_prompt], self.raw_prompts[idx // self.n_samples_per_prompt]
        else:
            return self.prompts[idx // self.n_samples_per_prompt]

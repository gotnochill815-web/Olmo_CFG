import random
import torch

class CFGDataCollator:

    def __init__(
        self,
        tokenizer,
        dropout_prob=0.2,
        max_length=512,
    ):

        self.tokenizer = tokenizer
        self.dropout_prob = dropout_prob
        self.max_length = max_length
        self.molstart_id = tokenizer.convert_tokens_to_ids("<molstart>")

    def build_prompt(self, sample):

        if random.random() < self.dropout_prob:

            return (
                "<NULL>\n"
                f"<molstart>{sample['smiles']}"
            )

        return (
            "<pstart>\n"
            f"<QED> {sample['qed']:.4f}\n"
            f"<LOGP> {sample['logp']:.4f}\n"
            f"<TPSA> {sample['tpsa']:.4f}\n"
            f"<SAS> {sample['sas']:.4f}\n"
            f"<molstart>{sample['smiles']}"
        )

    def __call__(self, batch):

        prompts = [
            self.build_prompt(sample)
            for sample in batch
        ]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = encoded["input_ids"].clone()

        # Ignore padding
        labels[encoded["attention_mask"] == 0] = -100

        for i in range(labels.size(0)):

            input_ids = encoded["input_ids"][i]

            mol_positions = (input_ids == self.molstart_id).nonzero(as_tuple=True)[0]

            if len(mol_positions) > 0:

                mol_pos = mol_positions[0].item()

                # Ignore everything up to and including <molstart>
                labels[i, :mol_pos + 1] = -100

        encoded["labels"] = labels

        return encoded

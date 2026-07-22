"""
CFG Data Collator
Supports:
- all property conditioning
- single property conditioning
- pair conditioning
- triple conditioning
- random conditioning
"""

import random
import torch


class CFGDataCollator:

    def __init__(
        self,
        tokenizer,
        max_length=512,
        dropout_prob=0.2,
        conditioning_mode="all",
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dropout_prob = dropout_prob
        self.conditioning_mode = conditioning_mode

    ###########################################################
    # Prompt Builder
    ###########################################################

    def build_prompt(
        self,
        smiles,
        qed=None,
        logp=None,
        tpsa=None,
        sas=None,
    ):

        prompt = "<pstart>\n"

        if qed is not None:
            prompt += f"<QED> {qed:.2f}\n"

        if logp is not None:
            prompt += f"<LOGP> {logp:.2f}\n"

        if tpsa is not None:
            prompt += f"<TPSA> {tpsa:.2f}\n"

        if sas is not None:
            prompt += f"<SAS> {sas:.2f}\n"

        prompt += "<molstart>"
        prompt += smiles

        return prompt

    ###########################################################
    # Choose conditioning
    ###########################################################

    def choose_properties(
        self,
        qed,
        logp,
        tpsa,
        sas,
    ):

        props = {
            "qed": qed,
            "logp": logp,
            "tpsa": tpsa,
            "sas": sas,
        }

        names = list(props.keys())

        #######################################################
        # ALL
        #######################################################

        if self.conditioning_mode == "all":
            return props

        #######################################################
        # SINGLE
        #######################################################

        if self.conditioning_mode == "single":

            keep = random.choice(names)

            return {
                k: props[k] if k == keep else None
                for k in names
            }

        #######################################################
        # PAIR
        #######################################################

        if self.conditioning_mode == "pair":

            keep = random.sample(names, 2)

            return {
                k: props[k] if k in keep else None
                for k in names
            }

        #######################################################
        # TRIPLE
        #######################################################

        if self.conditioning_mode == "triple":

            keep = random.sample(names, 3)

            return {
                k: props[k] if k in keep else None
                for k in names
            }

        #######################################################
        # RANDOM
        #######################################################

        if self.conditioning_mode == "random":

            mode = random.choice(
                [
                    "single",
                    "pair",
                    "triple",
                    "all",
                ]
            )

            self.conditioning_mode = mode

            selected = self.choose_properties(
                qed,
                logp,
                tpsa,
                sas,
            )

            self.conditioning_mode = "random"

            return selected

        raise ValueError(
            f"Unknown conditioning mode: {self.conditioning_mode}"
        )

    ###########################################################
    # Collator
    ###########################################################

    def __call__(self, batch):

        prompts = []

        for sample in batch:

            props = self.choose_properties(
                sample["qed"],
                sample["logp"],
                sample["tpsa"],
                sample["sas"],
            )

            prompt = self.build_prompt(
                smiles=sample["smiles"],
                qed=props["qed"],
                logp=props["logp"],
                tpsa=props["tpsa"],
                sas=props["sas"],
            )

            prompts.append(prompt)

        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = tokens["input_ids"].clone()

        molstart_id = self.tokenizer.convert_tokens_to_ids("<molstart>")

        for i in range(labels.size(0)):

            idx = (tokens["input_ids"][i] == molstart_id).nonzero()

            if len(idx) > 0:

                idx = idx[0].item()

                labels[i, : idx + 1] = -100

            labels[i][tokens["attention_mask"][i] == 0] = -100

        tokens["labels"] = labels

        return tokens
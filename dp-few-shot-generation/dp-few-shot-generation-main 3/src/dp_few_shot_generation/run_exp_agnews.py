# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import math
import re
import sys
import time
import traceback
import tqdm
from collections.abc import Iterable, Set
from typing import Annotated, cast, Optional
import aiohttp
import more_itertools
import numpy as np
import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
import os
from dotenv import load_dotenv
from openai.types import Completion
from openai import OpenAI
from openai._exceptions import BadRequestError, APIStatusError


load_dotenv()
client = OpenAI(api_key="your-api-key")
client = OpenAI()
prompt = "Hello, world!"       # str
l = 5                          # int
temp = 0.7                     # float
num_log_probs = 5              # int or None
echo = False                   # bool
n = 1
try:
    response = client.completions.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=l,
    temperature=temp,
    logprobs=num_log_probs,
    echo=echo,
    n=n,
)
    print(response)
except BadRequestError as e:
    print("Bad request:", e)
except APIStatusError as e:
    print("API status error:", e)

import scipy.special
import tqdm
import typer
from datasets import DatasetDict, load_dataset
from lmapi.lm import LM, CompletionsSettings
from lmapi.openai import client_session

from dp_few_shot_generation.lm import (
    api_openai_com,
    next_logprobs,
    normalized_logprobs_for_chosen_tokens,
)
from dp_few_shot_generation.prob_utils import densify, log_max_normalize, log_normalize

DEFAULT_NUM_PRIVATE_TRAIN = 20
DEFAULT_NUM_PUBLIC_TRAIN = 0
DEFAULT_NUM_VALID = 4
DEFAULT_NUM_PRIVATE_TRAIN_SPLITS = 10
DEFAULT_NUM_TEST = 1000

labels = ["World", "Sport", "Business", "Technology"]
label_dict = {0: ["World"], 1: ["Sports"], 2: ["Business"], 3: ["Technology"]}


def format_full_datum_for_prompt(labels, datum: dict[str, str]):
    return f'News Type: "{labels[datum["label"]]}"\nText: "{datum["text"] + " END"}"\n'


def format_test_input_for_prompt(labels, test_input: int):
    return f'News Type: "{labels[test_input]}"\nText: "'


def construct_prompt_same(train_examples, test_example):
    prompt = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
    for train_example in train_examples:
        prompt += "Article: " + train_example["text"] + "\n"
        prompt += "Answer: " + label_dict[train_example["label"]][0] + "\n\n"
    prompt += "Article: " + test_example["text"] + "\n"
    prompt += "Answer:"
    return prompt


def complete(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=l,
        temperature=temp,
        logprobs=num_log_probs,
        echo=echo,
        n=n,
    )
    return response

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_model_response(data, test_examples, openai_model):
    all_raw_answers = []

    prompts = []
    train_examples = data

    for test_example in test_examples:
        prompts.append(construct_prompt_same(train_examples, test_example))

    chunked_prompts = list(chunks(prompts, 20))
    for test_chunk in chunked_prompts:
        response = complete(test_chunk, l=1, model_name=openai_model, num_log_probs=100)
        response = cast(ChatCompletion, response)
        # Access .choices instead of response["choices"]
        for answer in response.choices:
            all_raw_answers.append(answer)

    return all_raw_answers



def get_label_probs(all_raw_answers, test_subset):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(label_dict)
    approx = False
    assert len(all_raw_answers) == len(test_subset)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    cnt = 0
    for i, ans in enumerate(all_raw_answers):
        try:
            top_logprobs = ans["logprobs"]["top_logprobs"][
                0
            ]  # [0] since we only ask for complete one more token
        except:
            cnt += 1  # cnt for corner case
        label_probs = [0] * len(label_dict.keys())
        top_logprobs = ans["logprobs"]["top_logprobs"][0]
        for j, label_list in label_dict.items():
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = " " + label  # notice prompt does not have space after 'A:'
                if label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                else:
                    all_found = False
            if not all_found:
                position = (i, j)  # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs)  # prob not normalized

    return all_label_probs  # NOT NORMALIZED


def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        if np.sum(label_probs) > 0:  # corner case np.sum(label_probs)=0.
            label_probs = label_probs / np.sum(label_probs)  # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)


def get_p_content_free(train_subset, openai_model, content_free_inputs=("N/A",)):
    """Query model with content-free input, return its prediction probability for each label"""
    all_p_y = []
    for content_free_input in content_free_inputs:
        prompt = construct_prompt_same(train_subset, content_free_input)
        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                response = complete(
                prompt + " " + a,
                l=0,
                model_name=openai_model,
                echo=True,
                num_log_probs=1
                )
                response: Completion = response
                # ✅ Access with dot notation
                logprobs = response.choices[0].logprobs
                if logprobs is not None and logprobs.token_logprobs:
                    logprob = logprobs.token_logprobs[-1]
                else:
                    logprob = float("-inf")  # Or skip this one, or set to 0

                prob += np.exp(logprob)
            p_y[i] = prob
        all_p_y.append(p_y)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y)  # normalize
    return p_y



def merge_logprobs_topk_mean(
    private_next_logprobs: list[dict[int, float]],
    public_next_logprobs: Optional[dict[int, float]],
    n_vocab: int,
    no_public_token: bool,
    normalize_max: bool,
) -> np.ndarray:
    normalize_fn = log_max_normalize if normalize_max else log_normalize

    if no_public_token:
        mats = [
            densify(n_vocab, normalize_fn(lps))
            for lps in private_next_logprobs
        ]
        merged = scipy.special.logsumexp(np.stack(mats), axis=0) - np.log(len(mats))
    else:
        assert public_next_logprobs is not None
        mats = [
            densify(
                n_vocab,
                normalize_fn({k: v for k, v in lps.items() if k in public_next_logprobs}),
            )
            for lps in private_next_logprobs
        ]
        merged = scipy.special.logsumexp(np.stack(mats), axis=0) - np.log(len(mats))
    return np.exp(merged)


async def generate_with_private_prompts(
    trainset,
    num_private_train: int,
    num_private_train_splits: int,
    instruction: str,
    public_train_prompt: str,
    stop_tokens: set[int],
    test_input: int,
    lm: LM,
    noise_rng: np.random.RandomState,
    sigma: float,
    labels: list[str],
    top_p: float,
    no_public_token: bool,
    subsample_per_token: bool,
    sample_same_label_prompts: bool,
    gen_seed: int,
    max_tokens: int = 99,
    normalize_max: bool = False,
) -> list[int]:
    """
    Generate tokens privately (and optionally publicly) with DP noise added.
    """
    generated: list[int] = []
    # prepare public prompt
    test_str = format_test_input_for_prompt(labels, test_input)
    public_prompt = public_train_prompt + test_str
    public_ids = lm.encoding.encode(public_prompt)

    # filter trainset if needed
    if sample_same_label_prompts:
        idxs = [i for i, d in enumerate(trainset) if d["label"] == test_input]
        train_subset = trainset.select(idxs)
    else:
        train_subset = trainset

    # initial private prompts
    private_ids_list: list[list[int]] = []
    if not subsample_per_token:
        subset = train_subset.shuffle(gen_seed).select(range(num_private_train))
        splits = list(more_itertools.distribute(num_private_train_splits, subset))
        private_prompts = [
            instruction + "\n".join(
                format_full_datum_for_prompt(labels, d) for d in split
            ) + test_str
            for split in splits
        ]
        private_ids_list = [lm.encoding.encode(p) for p in private_prompts]

    count = 0
    for _ in tqdm.tqdm(range(max_tokens), total=float("inf"), unit="token"):
        # rebuild private prompts each token if required
        if subsample_per_token:
            subset = train_subset.shuffle(gen_seed + count).select(range(num_private_train))
            splits = list(more_itertools.distribute(num_private_train_splits, subset))
            private_prompts = [
                instruction + "\n".join(
                    format_full_datum_for_prompt(labels, d) for d in split
                ) + test_str
                for split in splits
            ]
            private_ids_list = [lm.encoding.encode(p) for p in private_prompts]
            count += 1

        if no_public_token:
            # purely private mechanism
            private_next = await asyncio.gather(
                *(next_logprobs(lm, ids + generated, top_p) for ids in private_ids_list)
            )
            merged = merge_logprobs_topk_mean(private_next, None, lm.encoding.n_vocab, True, normalize_max)
            noise = (
                noise_rng.exponential(scale=sigma, size=lm.encoding.n_vocab)
                if normalize_max
                else noise_rng.normal(0, sigma, lm.encoding.n_vocab)
            )
            merged += noise
        else:
            # public + private
            public_next = await next_logprobs(lm, public_ids + generated, top_p)
            private_next = await asyncio.gather(
                *(normalized_logprobs_for_chosen_tokens(
                    lm, ids + generated, public_next.keys(), top_p
                ) for ids in private_ids_list)
            )
            merged = merge_logprobs_topk_mean(private_next, public_next, lm.encoding.n_vocab, False, normalize_max)
            noise = (
                noise_rng.exponential(scale=sigma, size=len(public_next))
                if normalize_max
                else noise_rng.normal(0, sigma, len(public_next))
            )
            merged[list(public_next.keys())] += noise

        next_id = int(np.argmax(merged))
        if next_id in stop_tokens:
            break
        generated.append(next_id)

    return generated



async def generate_with_public_prompt(
    public_train_prompt: str,
    stop_tokens: Set[str],
    test_input: str,
    lm: LM,
    labels,
    max_tokens: int = 500,
) -> list[int]:
    public_prompt = public_train_prompt + format_test_input_for_prompt(
        labels, test_input
    )
    public_prompt_tokens = lm.encoding.encode(public_prompt)
    public_prompt_tokens = public_prompt

    [completion] = await lm.completions(
        public_prompt_tokens,
        CompletionsSettings(
            temperature=0.0, max_tokens=max_tokens, n=1, stop=list(stop_tokens)
        ),
    )
    generated_tokens = [st.token.token_id for st in completion]
    return generated_tokens


def select_uniform_n_shots_over_labels(data, n_shots):
    select_list = []
    n_shots_per_label = math.ceil(n_shots / len(labels))
    labels_counter = {label[1][0]: n_shots_per_label for label in label_dict.items()}
    n_shots_selected = 0
    for i in range(len(data)):
        label = label_dict[data[i]["label"]][0]
        if labels_counter[label] == 0:
            continue
        else:
            labels_counter[label] -= 1
            select_list.append(i)
            n_shots_selected += 1
        if n_shots_selected == n_shots:
            break
    query_subset = data.select(select_list)
    return query_subset


def _main(
    sigma: Annotated[float, typer.Option()],  # noise parameters
    openai_model: Annotated[str, typer.Option()] = "babbage",
    print_prompts: Annotated[bool, typer.Option()] = False,
    # num_private_train=MN. MN=0 with num_valid=4 will get epsilon=0 (4-shot) results.
    num_private_train: Annotated[int, typer.Option()] = DEFAULT_NUM_PRIVATE_TRAIN,
    # by default set to 0. set_num_public_train >0 indicates additional public data available.
    set_num_public_train: Annotated[int, typer.Option()] = DEFAULT_NUM_PUBLIC_TRAIN,
    # num_valid=n. n samples to be generated for n-shot ICL
    num_valid: Annotated[int, typer.Option()] = DEFAULT_NUM_VALID,
    # num_private_train_splits=M
    num_private_train_splits: Annotated[
        int, typer.Option()
    ] = DEFAULT_NUM_PRIVATE_TRAIN_SPLITS,
    num_test: Annotated[int, typer.Option()] = DEFAULT_NUM_TEST,
    # no_public_token=True, RVP=False; no_public_token=False, RVP=True
    no_public_token: Annotated[bool, typer.Option()] = False,
    # subsample_per_token=True: at each token generation, subsample a new test set
    subsample_per_token: Annotated[bool, typer.Option()] = False,
    use_dp_prompts: Annotated[bool, typer.Option()] = False,
    # sample_same_label_prompts=True: sample subsets from the sets with same targeted labels.
    sample_same_label_prompts: Annotated[bool, typer.Option()] = False,
    # normalize_max=True, Exponential mechanism; normalize_max=False, Gaussian mechanism
    normalize_max: Annotated[bool, typer.Option()] = False,
    # max_token_per_text=T_max
    max_token_per_text: Annotated[int, typer.Option()] = 100,
    # consistent with default parameters in the documentation https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#completions
    top_p: Annotated[float, typer.Option()] = 1,
    # random seed for subsampling in generation
    synth_seed: Annotated[int, typer.Option()] = 0,
    # random seed for n-shot demonstrations sampling in evaluation
    eval_seed: Annotated[int, typer.Option()] = 0,
):
    async def main():
        if (num_private_train == 0) != (num_private_train_splits == 0):
            raise ValueError(
                "Either both or neither of --num-private-train and --num-private-train-splits can be 0"
            )
        command = ["python", sys.argv[0]]
        for x in sys.argv[1:]:
            if x.startswith("--"):
                assert '"' not in x and "'" not in x
                command.append(x)
            else:
                assert "'" not in x
                if re.match("^[a-zA-Z0-9_]+$", x):
                    command.append("%s" % x)
                else:
                    command.append("'%s'" % x)
        command = " ".join(command)
        print(command)

        if no_public_token:
            num_public_train = 0
        else:
            num_public_train = set_num_public_train

        lm = api_openai_com(openai_model)
        noise_rng = np.random.RandomState()

        data = cast(DatasetDict, load_dataset("ag_news"))
        print(labels)

        trainset = data["train"].shuffle(seed=synth_seed)
        print("trainset length", len(trainset))
        if num_public_train > 0:
            public_train_subset = cast(
                Iterable[dict[str, str]],
                trainset.select(
                    range(
                        len(trainset) - num_public_train,
                        len(trainset),
                    )
                ),
            )
        else:
            public_train_subset = []

        trainset = trainset.select(
            range(len(trainset) - num_public_train)
        )
        queryset = data["train"].shuffle(seed=eval_seed)
        query_subset = select_uniform_n_shots_over_labels(queryset, num_valid)

        if use_dp_prompts:
            synthetic_examples = []

            # Turn the data into prompts
            instruction = "Given a label of news type, generate the chosen type of news accordingly.\n\n"

            public_train_prompt = instruction + "\n".join(
                format_full_datum_for_prompt(labels, datum)
                for datum in public_train_subset
            )

            if print_prompts:
                print(public_train_prompt)
                print("=========")

            if normalize_max:
                print("Exponential Mechanism")
                assert num_private_train == 0 or sigma > 0
                if num_private_train > 0:
                    # scale == sigma_calib == 1/lambda. lambda for exponential distribution.
                    sigma_calib = (2 / num_private_train_splits) * (1 / sigma)
            else:
                print("Gaussian Mechanism")
                if num_private_train_splits > 0:
                    sigma_calib = math.sqrt(2) / num_private_train_splits * sigma
                else:
                    sigma_calib = 0
            print(
                f"sigma in command {sigma}. sigma added according to sensitivity {sigma_calib}"
            )

            stop_tokens = {"\n", "<|endoftext|>", " END"}
            stop_tokens_ids = {lm.encoding.encode_single_token(t) for t in stop_tokens}

            client_session.set(aiohttp.ClientSession())

            async with client_session.get():
                for i, test_datum in enumerate(query_subset, 1):
                    print(f"# Example {i}")
                    print(f'News Type: "{labels[test_datum["label"]]}"')
                    print(f'References:\n "{test_datum["text"]}"')

                    np.random.seed(synth_seed + i)
                    gen_seed = np.random.randint(100000)
                    print(f"gen-seed: {gen_seed}")

                    if num_private_train_splits > 0:
                        generated_token_ids = await generate_with_private_prompts(
                            trainset,
                            num_private_train,
                            num_private_train_splits,
                            instruction,
                            public_train_prompt,
                            stop_tokens_ids,
                            test_datum["label"],
                            lm,
                            noise_rng,
                            sigma_calib,
                            labels,
                            top_p,
                            no_public_token,
                            subsample_per_token,
                            sample_same_label_prompts,
                            gen_seed,
                            max_tokens=max_token_per_text
                            - 1,  # need one token length for EOS.
                            normalize_max=normalize_max,
                        )
                    else:
                        generated_token_ids = await generate_with_public_prompt(
                            public_train_prompt,
                            stop_tokens,
                            test_datum["label"],
                            lm,
                            labels,
                            max_tokens=max_token_per_text,
                        )

                    generated = lm.encoding.decode(generated_token_ids).rstrip('"')

                    print(f"Generated: {generated}\n")
                    output_datum = {}
                    output_datum["text"] = generated.strip()
                    output_datum["label"] = test_datum["label"]
                    synthetic_examples.append(output_datum)

        if num_test > 0:
            test_subset = (
                data["test"]
                .shuffle(seed=12345)
                .select(range(num_test))
            )
            test_labels = [test_example["label"] for test_example in test_subset]

            content_free_inputs = [{"text": "N/A"}, {"text": ""}, {"text": "[MASK]"}]
            p_cf_wout_DP = get_p_content_free(
                query_subset, openai_model, content_free_inputs=content_free_inputs
            )

            all_raw_answers_wout_DP = get_model_response(
                query_subset, test_subset, openai_model
            )
            all_label_probs_wout_DP = get_label_probs(
                all_raw_answers_wout_DP, test_subset
            )

            acc_original_wout_DP = eval_accuracy(all_label_probs_wout_DP, test_labels)
            acc_calibrated_wout_DP = eval_accuracy(
                all_label_probs_wout_DP,
                test_labels,
                mode="diagonal_W",
                p_cf=p_cf_wout_DP,
            )

            print(f"Accuracy (original) without DP: {acc_original_wout_DP}")
            print(f"Accuracy (calibrated) without DP: {acc_calibrated_wout_DP}")

            if use_dp_prompts:
                p_cf_w_DP = get_p_content_free(
                    synthetic_examples,
                    openai_model,
                    content_free_inputs=content_free_inputs,
                )

                all_raw_answers_w_DP = get_model_response(
                    synthetic_examples, test_subset, openai_model
                )
                all_label_probs_w_DP = get_label_probs(
                    all_raw_answers_w_DP, test_subset
                )

                acc_original_w_DP = eval_accuracy(all_label_probs_w_DP, test_labels)
                acc_calibrated_w_DP = eval_accuracy(
                    all_label_probs_w_DP, test_labels, mode="diagonal_W", p_cf=p_cf_w_DP
                )

                print(f"Accuracy (original) with DP: {acc_original_w_DP}")
                print(f"Accuracy (calibrated) with DP: {acc_calibrated_w_DP}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    typer.run(_main)

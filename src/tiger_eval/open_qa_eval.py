#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, May 14th 2024, 10:54:48 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import torch
import transformers

from tqdm import tqdm


def score(model_path, input_data):
    """ Compute the score of the model on the given data."""

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, device_map="auto", use_fast=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    model.eval() # set to eval mode, by default it is in eval model but just in case

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # generation
    all_details = []
    questions, references, predictions = input_data

    for question, reference, prediction in tqdm(zip(questions, references, predictions), total=len(questions)):

        PROMPT_TEMPLATE = """\
            [Question]
            {question}

            [Assistant Response]
            {prediction}

            [Ground Truth Response]
            {reference}

            [System]
            Please rate the accuracy of the assistant response based on its alignment with the ground truth response. Your rating should strictly reflect factual correctness:
            - If the ground truth directly contradicts the assistant's response, the rating must reflect an incorrect answer.
            - If the ground truth is ambiguous or states 'cannot decide', please rate it as incorrect.
            - Ratings should range from 1 to 5, where 1 indicates no factual alignment (incorrect) and 5 indicates complete factual alignment (correct).
            Your response should be formatted as follows:
            Explanation: (Provide a brief explanation of why the rating was assigned, focusing on factual correctness.)
            Rating: (int)"""
        
        evaluation_prompt = PROMPT_TEMPLATE.format(question=question, prediction=prediction, reference=reference)

        messages = [
            {"role": "system", "content": "You are an expert grader!"},
            {"role": "user", "content": evaluation_prompt},
        ]

        templated_sample = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False,
        )

        encoded_sample = tokenizer(templated_sample, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **encoded_sample,
            max_new_tokens=500,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
            do_sample=False,
        )

        response = outputs[0][encoded_sample.input_ids.shape[-1]:]
        output   = tokenizer.decode(response, skip_special_tokens=True).strip()

        try:
            rate_score = float(output.split()[-1])
            success = 1
        except:
            rate_score = 1.0
            success = 0

        sample_rating_detail = {
            'question'        : question,
            'reference'       : reference,
            'model_prediction': prediction,
            'judge_response'  : output,
            'rate_score'      : rate_score,
            'success'         : success,
        }

        all_details.append(sample_rating_detail)

    all_scores   = [detail['rate_score'] for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores)
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate, 'details': all_details}

    return judge_results


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
            Rate whether the assistant response correctly matches the ground truth.
            The rating should be 1-5, where 1 is incorrect and 5 is correct.
            Your response should be in the format:
            Explanation: (your explanation)
            Rating: (int)"""
        
        evaluation_prompt = PROMPT_TEMPLATE.format(question=question, prediction=prediction, reference=reference)

        #sample_input = "Check whether the generated answer has exactly the same meaning to the reference answer. Respond with 'Yes' or 'No'.\n\nQuestion:\n{}\n\nReference Answer:\n<<{}>>\n\nGenerated Answer:\n<<{}>>\n\n".format("", reference, prediction)

        # sample_input = "Check whether the generated answer has the same meaning as the reference answer. Respond with 'Yes' or 'No'.Check whether the generated answer is correct for recognizing the gender of the speaker. The groundtruth is stated in the referene answer. Rate as incorrect if the gender does not match or not stated. Respond with 'Correct' or 'Incorrect'.\n\nQuestion:\n{}\n\nReference Answer:\n<<{}>>\n\nGenerated Answer:\n<<{}>>\n\n".format(question, reference, prediction)


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
        # templated_sample = templated_sample + "Response: "
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
            rate_score = 0.0
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

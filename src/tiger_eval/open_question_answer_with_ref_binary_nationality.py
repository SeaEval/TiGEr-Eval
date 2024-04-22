
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

        sample_input = "Check whether the generated answer has the same nationality recognized to the reference answer. e.g. American = USA, UK != USA. Respond with 'Yes' or 'No'.\n\nQuestion:\n{}\n\nReference Answer:\n<<{}>>\n\nGenerated Answer:\n<<{}>>\n\n".format("", reference, prediction)

        # sample_input = "Check whether the generated answer has the same meaning as the reference answer. Respond with 'Yes' or 'No'.Check whether the generated answer is correct for recognizing the gender of the speaker. The groundtruth is stated in the referene answer. Rate as incorrect if the gender does not match or not stated. Respond with 'Correct' or 'Incorrect'.\n\nQuestion:\n{}\n\nReference Answer:\n<<{}>>\n\nGenerated Answer:\n<<{}>>\n\n".format(question, reference, prediction)


        messages = [
            {"role": "system", "content": "You are an expert grader! Please rate the generated answer."},
            {"role": "user", "content": sample_input},
        ]

        templated_sample = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False,
        )
        templated_sample = templated_sample + "Response: "
        encoded_sample = tokenizer(templated_sample, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **encoded_sample,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
            #do_sample=True,
            #temperature=0.6,
            #top_p=0.9,
        )
        response = outputs[0][encoded_sample.input_ids.shape[-1]:]
        output   = tokenizer.decode(response, skip_special_tokens=True).strip()


        try:
            if output.lower().startswith('yes'): rate_score = 1.0
            elif output.lower().startswith('no'): rate_score = 0.0
            else: raise ValueError("Invalid rating")
            success = 1
        except:
            rate_score = 0.0
            success = 0

        sample_rating_detail = {
            'question'        : question,
            'reference'       : reference,
            'model_prediction': prediction,
            'rate_score'      : rate_score,
            'success'         : success,
        }

        all_details.append(sample_rating_detail)

    all_scores   = [detail['rate_score'] for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores)
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    results = {'llm_score': avg_score, 'success_rate': success_rate, 'details': all_details}

    return results

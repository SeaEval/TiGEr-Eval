
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

        sample_input = "You will be given the question and the reference answer. Please rate the correctness of the generated answer from 1(worst) to 10(best).\n\nQuestion:\n{}\n\nReference Answer:\n{}\n\nGenerated Answer:\n{}\n\n".format(question, reference, prediction)

        messages = [
            {"role": "system", "content": "You are an expert grader who consistently evaluates on a scale from 1(worst) to 10(best)!"},
            {"role": "user", "content": sample_input},
        ]

        templated_sample = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=False,
        )
        templated_sample = templated_sample + "My rating score is: "
        encoded_sample = tokenizer(templated_sample, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **encoded_sample,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
            #do_sample=True,
            #temperature=0.6,
            #top_p=0.9,
        )
        response = outputs[0][encoded_sample.input_ids.shape[-1]:]
        output = tokenizer.decode(response, skip_special_tokens=True)

        try:
            rate_score = float(output.split()[0])
            success = 1
        except:
            rate_score = 0.0
            success = -1

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

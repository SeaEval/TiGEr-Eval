
import torch
import transformers

def llama_generate(model, tokenizer, batch_input):
    max_new_tokens = 100
    formatted_batch_input = []
    for input in batch_input:
        dialog = [{"role": "user", "content": input}]
        B_INST, E_INST = "[INST]", "[/INST]"
        formatted_batch_input.extend([f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"])
    batch_input = formatted_batch_input

    generation_config                = model.generation_config
    generation_config.max_new_tokens = max_new_tokens
    input_ids                        = tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, generation_config = generation_config)

    # remove the input_ids from the output_ids
    output_ids = output_ids[:, input_ids.shape[-1]:]
    outputs    = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs


def score(model_path, data_with_model_prediction):
    """ Compute the score of the model on the given data."""

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, device_map="auto", use_fast=False, padding_side='left')

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    model.eval() # set to eval mode, by default it is in eval model but just in case

    # Load Pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<unk>'})
        model.resize_token_embeddings(len(tokenizer))

    # generation
    all_scores = []
    for sample in data_with_model_prediction:
        batch_input = ["Please rate the quality (whether it is correct and helpful) of the generated answer from 1.0 to 5.0.\n\nQuestion:\n{}\n\nAnswer:\n{}\n".format(sample['question'], sample['model_prediction'])]
        eval_model_prediction = llama_generate(model, tokenizer, batch_input)[0]

        if any([item in eval_model_prediction for item in ['1.0', '2.0', '3.0', '4.0', '5.0']]):

            appeared_scores = [float(item) for item in ['1.0', '2.0', '3.0', '4.0', '5.0'] if item in eval_model_prediction]

            all_scores.append(max(appeared_scores))
        else:
            all_scores.append(1.0)

    results = {'llama_score': sum(all_scores) / len(all_scores)}

    return results

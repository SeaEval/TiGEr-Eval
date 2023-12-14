

import re
import random
from collections import Counter

index_to_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


def simple_segment(text):
        all_segments = []
        for sent in re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', text, flags=re.U):
            all_segments.append(sent)
        return all_segments

def replace_chinese_punctuation_with_english(text):
    chinese_punctuation = "，。“”‘’！？【】《》（）"
    english_punctuation = ",.\"\"''!?[]<>()"
    translation_table = str.maketrans(chinese_punctuation, english_punctuation)
    translated_string = text.translate(translation_table)
    return translated_string


def heuristic_align(choices, model_prediction):

    # if nothing in model_prediction, return random choice
    if len(model_prediction.strip()) == 0:
        return random.choice(choices)
    
    # In general, we take a majority voting approach, which we found to be more robust than a simple string matching approach.
    # First stage matching: match the first word of the model prediction to the first word of each choice
    number_of_choices    = len(choices)
    if number_of_choices > len(index_to_letter):
        raise ValueError('Number of choices is greater than the number of letters in index_to_letter, currently only supports up to 10 choices')
    
    first_stage_choices  = {}
    second_stage_choices = {}

    for i in range(number_of_choices):
        first_stage_choices[choices[i]]                                = choices[i]
        first_stage_choices[choices[i].lower()]                        = choices[i]
        first_stage_choices[choices[i].upper()]                        = choices[i]
        first_stage_choices["({})".format(index_to_letter[i])]         = choices[i]
        first_stage_choices["({})".format(index_to_letter[i].lower())] = choices[i]
        
        if index_to_letter[i] == 'A':
            second_stage_choices[" *&^%$asa "] = choices[i]
        else:
            second_stage_choices[" {} ".format(index_to_letter[i]).lower()] = choices[i]
        second_stage_choices[choices[i][4:].lower()] = choices[i]

    # filter out invalid candidates
    first_stage_choices      = {k: v for k, v in sorted(first_stage_choices.items(), key=lambda item: len(item[0]), reverse=True)}
    first_stage_choices_list = list(first_stage_choices.keys())

    second_stage_choices      = {simple_segment(k)[0]: v for k, v in sorted(second_stage_choices.items(), key=lambda item: len(item[0]), reverse=True)}
    second_stage_choices_list = list(second_stage_choices.keys())
    second_stage_choices_list = [item for item in second_stage_choices_list if len(item) > 1]

    model_prediction          = replace_chinese_punctuation_with_english(model_prediction)
    model_prediction_splitted = simple_segment(model_prediction)

    # all chosen candidates
    all_chosen_answers = []
    for sent in model_prediction_splitted:
        if not sent:
            continue

        chosen_answers = [item for item in first_stage_choices_list if item in sent] + [item for item in first_stage_choices_list if item in sent.lower()]

        if len(chosen_answers) == 0:
            continue

        # if more than one candidate is chosen, we choose the one with the longest length
        chosen_answers = sorted(chosen_answers, key=len, reverse=True)[0]
        all_chosen_answers.append(chosen_answers)

    if len(all_chosen_answers) != 0: # Perform majority voting
        chosen_labels = [first_stage_choices[item] for item in all_chosen_answers]
        counter = Counter(chosen_labels)
        top2 = counter.most_common(2)

        if len(top2) == 1:
            final_answer = top2[0][0]
        else:
            if top2[0][1] == top2[1][1]:
                final_answer = random.choice([top2[0][0], top2[1][0]])
            else:
                final_answer = top2[0][0]

        return final_answer

    else:
        # if no candidate is chosen, we use the second stage matching
        all_second_stage_chosen_answers = []
        for sent in model_prediction_splitted:
            if not sent:
                continue

            sent = sent.strip("(。$,|！|\!|\.|？|\?)，；：？！…—·《》“”‘’{}[]（）()、|\\/\n\t\r\v\f ")
            sent = " {} ".format(sent)
            sent = sent.replace(" A ", " *&^%$asa ").lower()

            second_stage_chosen_answers = [item for item in second_stage_choices_list if item in sent]

            if len(second_stage_chosen_answers) == 0:
                continue
            elif len(second_stage_chosen_answers) >= 1:
                all_second_stage_chosen_answers.append(second_stage_chosen_answers[0])
        
        if len(all_second_stage_chosen_answers) != 0:
            chosen_labels = [second_stage_choices[item] for item in all_second_stage_chosen_answers]
            counter = Counter(chosen_labels)
            top2 = counter.most_common(2)

            if len(top2) == 1:
                final_answer = top2[0][0]
            else:
                if top2[0][1] == top2[1][1]:
                    final_answer = random.choice([top2[0][0], top2[1][0]])
                else:
                    final_answer = top2[0][0]
            
            return final_answer
        
        # if no candidate is chosen, we return random choice
        else:
            return random.choice(choices)



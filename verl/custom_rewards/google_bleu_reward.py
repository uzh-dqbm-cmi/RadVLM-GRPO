import os
import re
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import random
import evaluate


MARK = "</think>"

google_bleu = evaluate.load("google_bleu")


def google_bleu_score(pred_text, true_text):
    result = google_bleu.compute(predictions=[pred_text], references=[[true_text]])
    return result["google_bleu"]
    

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "mimic_grpo":
        return google_bleu_score(solution_str, ground_truth)
    elif data_source == "mimic_grpo_reasoning":
        i = solution_str.rfind(MARK)

        if i == -1:
            return -0.0
        else:
            core_answer = solution_str[i + len(MARK):].strip()

            return google_bleu_score(core_answer, ground_truth)

    else:
        assert False, "incorrect data source."

if __name__ == "__main__":
    ref = "et tube terminates 2 cm above the carina retraction by several centimeters is recommended for more optimal placement bibasilar consolidations better assessed on concurrent chest ct"
    hyp = "endotracheal tube terminates 2 5 cm above the carina bibasilar opacities likely represent atelectasis or aspiration"
    score = compute_score(data_source="mimic_grpo", solution_str=hyp, ground_truth=ref)
    print(score)

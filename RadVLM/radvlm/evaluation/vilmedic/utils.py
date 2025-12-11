from __future__ import absolute_import
from . import *
from radgraph import F1RadGraph
from f1chexbert import F1CheXbert


def calcAllMetrics_by_one_by_one(target, prediction):
    """Calculate all metrics between target and prediction."""
    radgraph_simple, radgraph_partial, radgraph_complete = calcF1RadGraph(
        target, prediction
    )
    chexbert_all_micro, chexbert_all_macro, chexbert_5_micro, chexbert_5_macro = (
        calcChexbert(target, prediction)
    )
    # green_mean, greens = calcGREEN(target, prediction)
    return {
        "bleu": calcBLEU(target, prediction)[0],
        "bertscore": calcBertScore(target, prediction)[0],
        "meteor": calcMeteor(target, prediction)[0],
        "ciderd": calcCiderD(target, prediction)[0],
        "rouge1": calcRouge(target, prediction, rouges="ROUGE1")[0],
        "rouge2": calcRouge(target, prediction, rouges="ROUGE2")[0],
        "rougel": calcRouge(target, prediction, rouges="ROUGEL")[0],
        "radgraph_simple": radgraph_simple,
        "radgraph_partial": radgraph_partial,
        "radgraph_complete": radgraph_complete,
        "chexbert_all_micro": chexbert_all_micro,
        "chexbert_all_macro": chexbert_all_macro,
        "chexbert_5_micro": chexbert_5_micro,
        "chexbert_5_macro": chexbert_5_macro,
        # "green_mean": green_mean,
        # "greens": greens,
    }


def calcAllMetrics_whole(target_list, prediction_list):
    """Calculate all metrics between target and prediction."""
    bleu = calcBLEU(target_list, prediction_list)
    bert_score_average, bert_score_list = calcBertScore(target_list, prediction_list)
    #meteor = calcMeteor(target_list, prediction_list)
    ciderd = calcCiderD(target_list, prediction_list)
    rouge1 = calcRouge(target_list, prediction_list, rouges="ROUGE1")
    rouge2 = calcRouge(target_list, prediction_list, rouges="ROUGE2")
    rougel = calcRouge(target_list, prediction_list, rouges="ROUGEL")
    # set batch to False to give radgraph all at once, otherwise will load model for every inference.
    radgraph_simple, radgraph_partial, radgraph_complete = calcF1RadGraph(
        target_list, prediction_list, batch=False
    )
    chexbert_all_micro, chexbert_all_macro, chexbert_5_micro, chexbert_5_macro = (
        calcChexbert(target_list, prediction_list)
    )
    return {
        "blue": bleu[0],
        "bertscore": bert_score_average,
        #"meteor": meteor[0],
        "ciderd": ciderd[0],
        "rouge1": rouge1[0],
        "rouge2": rouge2[0],
        "rougel": rougel[0],
        "radgraph_simple": radgraph_simple,
        "radgraph_partial": radgraph_partial,
        "radgraph_complete": radgraph_complete,
        "chexbert_all_micro": chexbert_all_micro,
        "chexbert_all_macro": chexbert_all_macro,
        "chexbert_5_micro": chexbert_5_micro,
        "chexbert_5_macro": chexbert_5_macro,
    }


def calcBLEU(target, prediction):
    """Calculate BLEU score between target and prediction."""
    return Bleu()(target, prediction)


def calcBertScore(target, prediction):
    """Calculate BERTScore between target and prediction."""
    return BertScore()(target, prediction)


def calcMeteor(target, prediction):
    """Calculate METEOR score between target and prediction."""
    return Meteor()(target, prediction)


def calcCiderD(target, prediction):
    """Calculate CIDEr-D score between target and prediction."""
    return CiderD()(target, prediction)


def calcRouge(target, prediction, rouges="ROUNGE1"):
    """Calculate ROUGE score between target and prediction."""
    return Rouge(rouges=[rouges.lower()])(target, prediction)


def calcF1RadGraph(target, prediction, batch=False):
    """Calculate F1 score for RadGraph between target and prediction."""
    if batch:
        simple_list = []
        partial_list = []
        complete_list = []
        for t, p in zip(target, prediction):
            simple, partial, complete = F1RadGraph(
                reward_level="all", model_type="radgraph-xl"
            )([t], [p])[0]
            simple_list.append(simple)
            partial_list.append(partial)
            complete_list.append(complete)
        simple = sum(simple_list) / len(simple_list)  # average
        partial = sum(partial_list) / len(partial_list)
        complete = sum(complete_list) / len(complete_list)
        return simple, partial, complete
    else:
        max_t = 0
        max_p = 0
        max_t_str = None
        max_p_str = None
        for t, p in zip(target, prediction):
            curr_len_t = len(t)
            curr_len_p = len(p)
            if curr_len_t > max_t:
                max_t = curr_len_t
                max_t_str = t

            if curr_len_p > max_p:
                max_p = curr_len_p
                max_p_str = p

        print(f"{max_t=}\n{max_t_str=}\n{max_p=}\n{max_p_str=}")


        simple, partial, complete = F1RadGraph(
            reward_level="all", model_type="radgraph-xl"
        )(target, prediction)[0]
        return simple, partial, complete


def calcChexbert(target, prediction):
    """Calculate F1 score for CheXbert between target and prediction."""
    accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = F1CheXbert(
        refs_filename=None, hyps_filename=None
    )(prediction, target)
    return (
        chexbert_all["micro avg"]["f1-score"],
        chexbert_all["macro avg"]["f1-score"],
        chexbert_5["micro avg"]["f1-score"],
        chexbert_5["macro avg"]["f1-score"],
    )

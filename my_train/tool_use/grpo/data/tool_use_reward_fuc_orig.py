import re
import json
from typing import List, Dict
import math
from loguru import logger

def calc_name_jaccard(predictions: List[Dict], ground_truths: List[Dict]) -> float:
    pred_names = {entry["name"] for entry in predictions}
    gt_names = {entry["name"] for entry in ground_truths}

    intersection = pred_names & gt_names
    union = pred_names | gt_names

    if not union:
        return 1.0
    return len(intersection) / len(union)


def calc_param_key_jaccard(predictions: List[Dict], ground_truths: List[Dict]) -> float:
    total_score = 0.0
    count = 0

    for gt in ground_truths:
        gt_name = gt["name"]
        gt_keys = set(gt["arguments"].keys())

        best_match_score = 0
        for pred in predictions:
            if pred["name"] == gt_name:
                pred_keys = set(pred["arguments"].keys())
                union = gt_keys | pred_keys
                if not union:
                    match_score = 1.0
                else:
                    match_score = len(gt_keys & pred_keys) / len(union)
                best_match_score = max(best_match_score, match_score)

        total_score += best_match_score
        count += 1

    return total_score / count if count > 0 else 1.0


def calc_param_value_accuracy(predictions: List[Dict], ground_truths: List[Dict]) -> float:
    matched_values = 0
    total_gt_keys = 0

    for gt in ground_truths:
        gt_name = gt["name"]
        gt_args = gt["arguments"]

        best_pred = None
        for pred in predictions:
            if pred["name"] == gt_name:
                best_pred = pred
                break
        if best_pred is None:
            total_gt_keys += len(gt_args)
            continue

        pred_args = best_pred["arguments"]

        for key in gt_args:
            total_gt_keys += 1
            if key in pred_args and pred_args[key] == gt_args[key]:
                matched_values += 1

    if total_gt_keys == 0:
        return 1.0
    return matched_values / total_gt_keys


def max_score_per_sample(ground_truths: List[Dict]) -> float:
    return max(3 * len(ground_truths), 3.0)



def compute_score(
    data_source: dict,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    cur_step: int = 0,
    total_step: int = 0,
) -> float:
    print(f"当前Step: {cur_step} | {total_step}")

    # 检查格式
    format_pattern = r"<tool_call>\s*[\s\S]*?\s*</tool_call>\s*$"
    format_score = 1 if re.search(format_pattern, solution_str) else 0

    # 步数衰减
    if total_step > 0:
        ratio = cur_step / total_step
        if ratio > 0.3:
            x = (ratio - 0.3) / 0.7
            decay = math.exp(-5 * x)
            format_score *= decay
            print(f"当前Step的decay为: {decay}")

    # 提取 JSON
    json_pattern   = r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>"
    pred_json_strs = re.findall(json_pattern, solution_str, re.DOTALL)
    gt_json_strs = re.findall(json_pattern, ground_truth, re.DOTALL)

    try:
        predictions = [json.loads(i) for i in pred_json_strs]
        ground_truths = [json.loads(i) for i in gt_json_strs]
    except Exception:
        print("**json格式输出错误**")
        return -3

    # 各部分得分
    name_score = calc_name_jaccard(predictions, ground_truths)
    param_key_score = calc_param_key_jaccard(predictions, ground_truths)
    param_value_score = calc_param_value_accuracy(predictions, ground_truths)

    s_max = max_score_per_sample(ground_truths)
    total_score = name_score + param_key_score + param_value_score

    r_correct = 6 * (total_score / s_max) - 3
    final_score = format_score + r_correct
    return final_score


if __name__ == "__main__":
    pred_str = """<tool_call>
{"name": "create_folder", "arguments": {"accountId": 123, "folder_name": "游戏"}}
</tool_call>"""
    gt_str = """<tool_call>
{"name": "create_folder", "arguments": {"accountId": "123", "folder_name": "游戏"}}
</tool_call>"""
    score = compute_score(
        data_source={}, solution_str=pred_str, ground_truth=gt_str
    )
    print(score)

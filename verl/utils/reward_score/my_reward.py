import re
import json
from typing import List, Dict
import math


def compute_r_name(predict_jsons: List[Dict], ground_truth_jsons: List[Dict]) -> float:
    pred_names = {entry["name"] for entry in predict_jsons}
    gt_names = {entry["name"] for entry in ground_truth_jsons}

    intersection = pred_names & gt_names
    union = pred_names | gt_names

    if not union:
        return 1.0
    return len(intersection) / len(union)


def compute_r_param(predict_jsons: List[Dict], ground_truth_jsons: List[Dict]) -> float:
    total_score = 0.0
    count = 0

    for gt in ground_truth_jsons:
        gt_name = gt["name"]
        gt_keys = set(gt["arguments"].keys())

        best_match_score = 0
        for pred in predict_jsons:
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


def compute_r_value(predict_jsons: List[Dict], ground_truth_jsons: List[Dict]) -> float:
    matched_values = 0
    total_gt_keys = 0

    for gt in ground_truth_jsons:
        gt_name = gt["name"]
        gt_args = gt["arguments"]

        best_pred = None
        for pred in predict_jsons:
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


def compute_s_max(ground_truth_jsons):
    return 3 * len(ground_truth_jsons)


def compute_score(
    data_source: dict,
    solution_str,
    ground_truth,
    extra_info: dict = None,
    cur_step: int = 0,
    total_step: int = 0,
) -> float:
    print(f"Step进度: {cur_step} | {total_step}")
    pattern_format = r"<tool_call>\n[\s\S]*\n</tool_call>\s*$"
    if re.search(pattern_format, solution_str):
        format_score = 1
    else:
        format_score = 0

    if total_step > 0:
        ratio = cur_step / total_step
        if ratio > 0.3:
            x = (ratio - 0.3) / 0.7
            decay = math.exp(-5 * x)
            format_score *= decay
            print(f"decay为: {decay} | Format_score: {format_score}") 
    pattern_request = r"<tool_call>\n(.*?)\n</tool_call>"
    tool_request_json_str = re.findall(pattern_request, solution_str, re.DOTALL)
    ground_truth_json_str = re.findall(pattern_request, ground_truth, re.DOTALL)

    try:
        tool_request_json = [json.loads(i) for i in tool_request_json_str]
        ground_truth_json = [json.loads(i) for i in ground_truth_json_str]
    except Exception as e:
        print(f"**格式出错**")
        return -3

    name_score = compute_r_name(tool_request_json, ground_truth_json)
    param_score = compute_r_param(tool_request_json, ground_truth_json)
    value_score = compute_r_value(tool_request_json, ground_truth_json)
    s_max_score = compute_s_max(ground_truth_json)
    total_score = name_score + param_score + value_score
    r_correct = 6 * (total_score / s_max_score) - 3
    final_score = format_score + r_correct
    print(f"当前得分: {final_score}\n")

    return final_score


if __name__ == "__main__":
    predict_jsons = """<tool_call>
{"name": "create_folder", "arguments": {"accountId": "123", "folder_name": "游戏"}}
</tool_call>"""
    ground_truth_jsons = """<tool_call>
{"name": "create_folder", "arguments": {"accountId": "123", "folder_name": "游戏"}}
</tool_call>"""
    score = compute_score(
        data_source={}, solution_str=predict_jsons, ground_truth=ground_truth_jsons
    )
    print(score)

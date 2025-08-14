import re
import json
from typing import List, Dict
import math
from loguru import logger
from difflib import SequenceMatcher

def calc_name_jaccard(prediction: Dict, ground_truth: Dict) -> float:
    """
        工具调用 Name-Match 得分
        ** 传参形式: 由Str解析后的Dict
    """
    return 1.0 if prediction.get("name") == ground_truth.get("name") else 0.0

def calc_param_key_jaccard(prediction: Dict, ground_truth: Dict) -> float:
    """
        工具调用 ParamKeys-Match 得分
        ** 传参形式: 由Str解析后的Dict
    """
    gt_keys = set(ground_truth.get("arguments", {}).keys())
    pred_keys = set(prediction.get("arguments", {}).keys())

    union_keys = gt_keys | pred_keys
    if not union_keys: # 如果没有参数键，则认为完全匹配
        return 1.0
    else:
        keys_jaccard = len(gt_keys & pred_keys) / len(union_keys)
        return  keys_jaccard



def calc_param_value_jaccard(prediction: Dict, ground_truth: Dict) -> float:
    """
        工具调用 ParamValue-Jaccard 得分
        ** 传参形式: 由Str解析后的Dict
        "arguments": {
            "email_ids": [
            "a4b54fe8-855e-4ef1-9897-b7a7d130e45a",
            "ae9ba624-139e-42f2-963e-61d2dcae187b"
            ],
            "summary": "The selected emails have been successfully deleted and moved to the trash."
        }
    """
    gt_args = ground_truth.get("arguments", {})
    pred_args = prediction.get("arguments", {})
    gt_keys_num = len(gt_args)
    
    if not gt_keys_num: # Ground Truth 没有参数要求, 则任何 Pred 满足要求
        return 1.0
    
    total_score = 0.0
    for key, gt_val in gt_args.items():
        if key in pred_args:
            pred_val = pred_args[key]
            total_score += SequenceMatcher(None, str(pred_val), str(gt_val)).ratio()
        else:
            total_score += 0.0  # 缺键扣分

    return total_score / gt_keys_num



def max_score_per_sample(ground_truth: Dict) -> float:
    return 3.0  # 三项各∈[0,1]，上限为3


def compute_score(
    data_source: dict,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    cur_step: int = 0,
    total_step: int = 0,
) -> float:

    # 检查 tool_call 格式, 具有一票否决权
    format_pattern = r"<tool_call>\s*[\s\S]*?\s*</tool_call>\s*$"
    format_score = 1.0 if re.search(format_pattern, solution_str) else 0.0

    # 步数衰减
    if total_step > 0:
        ratio = cur_step / total_step
        if ratio > 0.3:
            x = (ratio - 0.3) / 0.7
            decay = math.exp(-5 * x)
            format_score *= decay
            logger.info(f"当前Step的decay为: {decay}")

    # 提取 JSON
    json_pattern   = r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>"
    pred_json_strs = re.findall(json_pattern, solution_str, re.DOTALL)
    gt_json_strs = re.findall(json_pattern, ground_truth, re.DOTALL)

    try:
        predictions = [json.loads(i) for i in pred_json_strs]
        ground_truths = [json.loads(i) for i in gt_json_strs]
    except Exception:
        logger.warning("**Json 格式解析失败**, 本次得分为 -3.0")
        return -3.0

    # 兼容上面评分函数的入参类型（Dict），取首个tool_call
    pred: Dict = predictions[0] if predictions else {}
    gt: Dict = ground_truths[0] if ground_truths else {}

    if not pred or not gt:
        logger.warning("**Dict 类型解析失败**, 本次得分为 -3.0")
        return -3.0

    # 各部分得分（保持原有计算逻辑）
    name_score = calc_name_jaccard(pred, gt) # 具有一票否决权
    param_key_score = calc_param_key_jaccard(pred, gt)
    param_value_score = calc_param_value_jaccard(pred, gt)

    s_max = max_score_per_sample(gt)
    # total_score = name_score + param_key_score + param_value_score
    total_score = name_score * (1 + param_key_score + param_value_score) # 只有 name_score 为 1 时，才会有 param_key_score 和 param_value_score 的加成

    r_correct = 6 * (total_score / s_max) - 3
    final_score = float(format_score + r_correct)
    logger.success(f"当前Step: {cur_step} | {total_step}\n格式分: {format_score:.2f} | Name分: {name_score:.2f} | ParamKey分: {param_key_score:.2f} | ParamValue分: {param_value_score:.2f} | 最终得分: {final_score:.2f}")
    return final_score


if __name__ == "__main__":
    pred_str = """<tool_call>
{"name": "焦点访谈", "arguments": {"accountId": "123", "folder_name": "游戏"}}
</tool_call>"""
    gt_str = """<tool_call>
{"name": "create_folder", "arguments": {"accountId": "123", "folder_name": "游戏"}}
</tool_call>"""
    score = compute_score(
        data_source={}, 
        solution_str = pred_str, 
        ground_truth = gt_str,
        cur_step = 75,
        total_step = 177,    
    )
    print(f"得分: {score}")

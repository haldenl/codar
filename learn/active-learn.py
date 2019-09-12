import json
import copy
from pprint import pprint

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import argparse

parser = argparse.ArgumentParser(description="codar active learning cli")

# data and label files
parser.add_argument('--data_file', type=str, default="example_pairs.json", help="the dataset pool")
parser.add_argument('--labels', type=str, default='{"1": ">", "2": ">", "3": ">", "4": ">", "5": ">", "6": ">", "7": ">", "8": ">", "9": ">", "10": ">", "11": ">", "12": ">", "13": ">", "14": ">", "15": ">", "16": ">"}', 
                    help="labels of the dataset: >, <, bad, unlabeled")

# active learning parameters
parser.add_argument('--min_label_size', type=int, default=15, help="the minimum number of labels we need to start active learning")
parser.add_argument('--sample_size', type=int, default=5, help="the number of samples will be presented for labeling at each time")
parser.add_argument('--sample_func', type=str, default="entropy", help="sample function, one of [entropy], [margin], [random]")
parser.add_argument('--prefer_low_complexity_pairs', type=bool, default=True, help="whether prefer low complexity pairs first")


def get_feature(chart):
    """Given a vis json object, parse its feature """
    def parse_soft_rule(rule):
        head = rule[:rule.index("(")]
        body = rule[rule.index("(") + 1: rule.index(")")]
        literals = body.strip().split(",")
        return head, literals

    all_facts = json.loads(chart["draco"]) if isinstance(chart["draco"], (str,)) else chart["draco"]
    facts = [x for x in all_facts if x.startswith("soft")]

    feature = {}
    for r in facts:
        h, l = parse_soft_rule(r)
        rname = l[0] + "-" + l[1]
        if rname not in feature:
            feature[rname] = 0
        feature[rname] += 1
    return feature

def get_pair_feature(pair):
    """Get the feature of a pair that will be used in active learning"""
    left_facts = get_feature(pair["left"])
    right_facts = get_feature(pair["right"])
    combined = copy.copy(left_facts)
    for k, v in right_facts.items():
        if k not in combined:
            combined[k] = 0
        combined[k] -= v
    return combined

def active_learn_one_iter(model, X_label, y_label, X_pool, sample_func, sample_size):
    """one iteration of active learning """
    model.fit(X_label, y_label)
    
    # different sample function to obtain new data
    if sample_func == "entropy":
        # entropy based sampling
        prob_vals = model.predict_proba(X_pool)
        entropy_uncertainty = (-prob_vals * np.log2(prob_vals)).sum(axis=1)
        selections = (np.argsort(entropy_uncertainty)[::-1])[:sample_size]
    elif sample_func == "margin":
        # entropy based sampling
        prob_vals = model.predict_proba(X_pool)
        rev = np.sort(prob_vals, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        selections = np.argsort(values)[:sample_size]
    elif sample_func == "random":
        selections = np.random.choice(list(range(len(X_pool))), sample_size)

    return selections

    
if __name__ == '__main__':
    args = parser.parse_args()

    # load dataset and labels
    data_file = args.data_file
    with open(data_file, "r") as f:
        dataset = json.load(f)

    labels = {int(pair_id): label for pair_id, label in json.loads(args.labels).items()}

    sample_func = args.sample_func
    sample_size = args.sample_size
    min_label_size = args.min_label_size

    features = { p["pair_id"]: get_pair_feature(p) for p in dataset }
    complexity = { p["pair_id"]: len(json.loads(p["left"]["draco"]) + json.loads(p["right"]["draco"])) for p in dataset }

    # get all features used by all tools
    all_features_names = list(set([k for p in features.values() for k in p.keys()]))
    
    # extract labeled and unlabelled
    labeled_data = [{"feature": feature, 
                     "label": labels[pair_id], 
                     "pair_id": pair_id} 
                     for pair_id, feature in features.items() 
                        if (pair_id in labels and labels[pair_id] in ["<", ">"])]

    pool_data = [{"feature": feature,
                  "pair_id": pair_id,
                  "complexity": complexity[pair_id]} for pair_id, feature in features.items() 
                 if (pair_id not in labels or labels[pair_id] == "unlabeled")]

    if len(labeled_data) < min_label_size:
        # when we don't have enough data to start active learning
        selections = np.random.choice(list(range(len(pool_data))), min_label_size - len(labeled_data))
        to_label_ids = [pool_data[k]["pair_id"] for k in selections]
    else:
        # the cross-validation size used for calibration
        calibration_cv_size = 5

        X_label = [[(p["feature"][feature_name] if feature_name in p["feature"] else 0) 
                            for feature_name in all_features_names] for p in labeled_data]
        y_label = [p["label"] for p in labeled_data]

        # flip some labels so that we get balanced x, y
        label_to_flip = "<" if y_label.count(">") < calibration_cv_size else (">" if y_label.count("<") < calibration_cv_size else None)
        if label_to_flip is not None:
            flip_count = calibration_cv_size - (len(y_label) - y_label.count(label_to_flip))
            for i in range(len(X_label)):
                if y_label[i] == label_to_flip:
                    X_label[i] = [-x for x in X_label[i]]
                    y_label[i] = ">" if y_label[i] == "<" else "<"
                    flip_count -= 1
                if flip_count <= 0:
                    break    

        X_label, y_label = np.array(X_label), np.array(y_label)

        X_pool = np.array([[(p["feature"][feature_name] if feature_name in p else 0) for feature_name in all_features_names] for p in pool_data])

        model = CalibratedClassifierCV(LinearSVC(fit_intercept=0), cv=calibration_cv_size)

        if args.prefer_low_complexity_pairs:
            selections = active_learn_one_iter(model, X_label, y_label, X_pool, sample_func, sample_size * 2)
            candidate_complexity = [pool_data[k]["complexity"] for k in selections]
            lowest_complexity_candidate = np.argsort(candidate_complexity)[:sample_size]
            # select samples with lower complexity
            selections = [selections[k] for k in lowest_complexity_candidate]
        else:
            selections = active_learn_one_iter(model, X_label, y_label, X_pool, sample_func, sample_size)

        to_label_ids = [pool_data[k]["pair_id"] for k in selections]

    #return to_label_ids
    print(json.dumps(to_label_ids))
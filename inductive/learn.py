import copy
import json
from pprint import pprint
import itertools
import sys

from vega import VegaLite

import data_utils

FACTS_HEAD = {
    "view_facts": ["mark"],
    "enc_facts": ["field_ty", "enc_ty", "channel", "aggr", "bin"]
}

META_VAR = {
    "@position_channel": ["x", "y"],
    "@mark_prop_channel": ["color", "size", "shape"],
    "@facet_channel": ["row", "column"],
    "@aggr": ["median", "sum", "max", "min", "avg"],
}


class Atom(object):
    def __init__(self, relation, terms):
        self.rel = relation
        self.terms = terms

    def __repr__(self):
        return f"{self.rel}({','.join([t for t in self.terms])})"

    def __eq__(self, other):
        if isinstance(other, Atom):
            if other.__repr__() == self.__repr__():
                return True
        return False

    def __hash__(self):
        return self.__repr__.__hash__()

class Rule(object):
    def __init__(self, head, body):
        self.head = head
        self.body = body

    def __repr__(self):
        return f"{self.head} :- {','.format([at for at in self.body])}"

    def __eq__(self, other):
        if isinstance(other, Rule):
            if other.__repr__() == self.__repr__():
                return True
        return False

    def __hash__(self):
        return self.__repr__.__hash__()

def intersect(l_lists):
    res = l_lists[0]
    for l in l_lists:
        res = [x for x in res if x in l]
    return res

def analyze():
    all_rels = {}
    for i,e in enumerate(vis_specs):
        props = infer_facts(e["vl"], [x for x in e["draco"] if not x.startswith("soft")]) 
        rels = infer_relations(props)
        #print(f"{i+1}-------------")
        #VegaLite(e["vl"], target_data).display()
        for r in rels:
            if r not in all_rels:
                all_rels[r] = {"p": 0, "n": 0}
            all_rels[r]["p" if labels[i] == True else "n"] += 1
    
    all_rels = [(key, val["n"], val["p"]) for key, val in all_rels.items()]
    
    pprint(sorted(all_rels, reverse = True, key=lambda x: x[1] - x[2]))
    #return target_data, labels, vis_specs 


def infer_facts(vl_spec, data_schema):
    """given a vis spec, infer facts for it """
    facts = []
    facts.append(Atom("mark", [vl_spec['mark']]))
    for i, key in enumerate(vl_spec["encoding"]):
        enc = vl_spec["encoding"][key]

        facts.append(Atom("channel", [key]))
        facts.append(Atom("enc_type", [key, enc["type"]]))

        if "field" in enc:
            field_type = data_schema[enc['field']]
            facts.append(Atom("field_type", [key, field_type]))

        if "aggregate" in enc:
            facts.append(Atom("aggregate", [key, enc['aggregate']]))
        
        if "bin" in enc and enc["bin"]:
            facts.append(Atom("bin", key))

    return facts

def canonicalize(rbody):
    # canonicalize a rule body
    return sorted(rbody, key=lambda at: (at.terms[0], at.rel))

def filter_invalid(rbody):
    # filter invalid rule bodies
    channels = [f.terms[0] for f in rbody if f.rel in ["enc_type", "field_type", "aggregate", "bin", "channel"]]
    for c in channels:
        if channels.count(c) == 1:
            return False
    return True

def lift_rule_body(rbody):
    # abstract a rule body with meta variables
    terms = list(set([t for fact in rbody for t in fact.terms]))
    mp, inv_map = {}, {}
    for t in terms:
        for var in META_VAR:
            if t in META_VAR[var]:
                if var not in mp:
                    mp[var] = []
                mp[var].append(t)
                inv_map[t] = f"{var}_{mp[var].index(t)}"
    abs_rbody = [Atom(fact.rel, [inv_map[t] if t in inv_map else t for t in fact.terms]) for fact in rbody]
    return abs_rbody

def gen_alpha_equivalent_rbody(rbody):
    # given a rule body, generate all other possible equivalent rule bodies
    terms = list(set([t for fact in rbody for t in fact.terms]))
    return rbody

def enumerate_rules(facts, size):
    if size == 1:
        return [[f] for f in facts]
    bases = enumerate_rules(facts, size - 1)
    rule_bodies = []
    for base in bases:
        used_terms = list(set([t for f in base for t in f.terms]))
        for f in facts:
            if f in base: 
                continue
            # remove the rule if the channel is not yet mentioned
            if f.rel in ["enc_type", "field_type", "aggregate", "bin"] and f.terms[0] not in used_terms: 
                continue
            rule_bodies.append(base + [f])
    return rule_bodies

def infer_rules(facts, max_size=5):
    """Given properties of a spec, infer relations over the spec"""
    rules = []

    head = Atom("invalid", ["v"])

    rule_bodies = [x for l in [enumerate_rules(facts, max_size) for max_size in range(1, max_size + 1) ]for x in l]
    rule_bodies = [canonicalize(rb) for rb in rule_bodies if filter_invalid(rb)]

    # infer abstract rule bodies
    abs_rule_bodies = {}
    for rbody in rule_bodies:
        print(rbody)
        print(lift_rule_body(rbody))

    #pprint(rule_bodies)
    #print(len(rule_bodies))

    sys.exit(-1)

            
    for size in range(2, max_size + 1):
        for lst in itertools.combinations(facts, size):
            view_facts, enc_facts = [], []
            for fact in lst:
                hd, args = parse_fact(fact)
                if hd in FACTS_HEAD["view_facts"]:
                    view_facts.append(fact)
                if hd in FACTS_HEAD["enc_facts"]:
                    enc_facts.append(fact)

            if len(enc_facts) > 0:
                common_literals = intersect([parse_fact(fact)[1] for fact in enc_facts])
                if len(common_literals) <= 1:
                    continue

            rules.append(f"invalid(v) :- {','.join([s for s in lst])}")
    return rules
    
def eliminate_pos(rels, pos_rels):
    return [x for x in rels if x not in pos_rels]

def main():
    synth_data = data_utils.load_synthetic_dataset()
    codar_data = data_utils.load_codar_dataset()

    for d in synth_data + synth_data:
        facts = infer_facts(d["vl"], d["data_schema"]) 
        rels = infer_rules(facts)
        pprint(facts)
        #pprint(rels)

    #pprint(loaded_data)

if __name__ == '__main__':
    main()

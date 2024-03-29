import copy
import json
from pprint import pprint
import itertools
import sys

from vega import VegaLite

import data_utils

RELATION_TYPE = {
    "view_facts": ["mark"],
    "enc_facts": ["field_ty", "enc_ty", "channel", "aggr", "bin"]
}

META_VAR = {
    "@CH":  ["bar", "text", "line", "point", "x", "y", "color", "size", "shape", "text", "row", "column", "detail"],
    # "@MK": ["bar", "text", "line", "point"],
    # "@PC": ["x", "y"], #position_channel
    # "@MC": ["color", "size", "shape", "text"], #mark_prop_channel
    # "@FC": ["row", "column"], #facet_channel
    "@AG": ["median", "sum", "max", "min", "avg", "count"], #aggr
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
        return self.__repr__().__hash__()

class Rule(object):
    def __init__(self, head, body):
        self.head = head
        self.body = body

    def __repr__(self):
        return f"{self.head} :- {','.join([str(at) for at in self.body])}"

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
    return tuple(sorted(rbody, key=lambda at: (at.terms[0], at.rel)))

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
    abs_rbody = tuple([Atom(fact.rel, [inv_map[t] if t in inv_map else t for t in fact.terms]) if fact.rel != "mark" else fact for fact in rbody])
    abs_rbody = canonicalize(abs_rbody)
    return abs_rbody

def gen_alpha_equivalent_rbody(rbody):
    # given a rule body, generate all other possible equivalent rule bodies

    meta_terms = list(set([t for fact in rbody for t in fact.terms if t.startswith("@")]))
    mp = {}
    for meta_var in META_VAR:
        for t in meta_terms:
            if t.startswith(meta_var):
                if meta_var not in mp:
                    mp[meta_var] = []
                mp[meta_var].append(t)
    
    # performing alpha-renamings
    alternative_maps = [{}]
    for meta_var in mp:
        srcs = mp[meta_var]
        updated_alternative_maps = []
        for dsts in itertools.permutations(srcs):
            for alter_map in alternative_maps:
                new_mp = copy.deepcopy(alter_map)
                for i in range(len(srcs)):
                    new_mp[srcs[i]] = dsts[i]
                updated_alternative_maps.append(new_mp)
        alternative_maps = updated_alternative_maps

    if len(alternative_maps) <= 1:
        return [rbody]

    all_equivs = []
    for alter_map in alternative_maps:
        new_rbody = tuple([Atom(fact.rel, [alter_map[t] if t in alter_map else t for t in fact.terms]) for fact in rbody])
        all_equivs.append(new_rbody)

    return all_equivs


def enumerate_rules(facts, size):
    """Enumerate rules from known facts """

    def enumerate_rule_body(facts, size):
        if size == 1:
            return [(f,) for f in facts]
        bases = enumerate_rule_body(facts, size - 1)
        rule_bodies = []
        for base in bases:
            used_terms = set([t for f in base for t in f.terms])
            for f in facts:
                if f in base: 
                    continue
                # remove the rule if the channel is not yet mentioned
                if f.rel in ["enc_type", "field_type", "aggregate", "bin"] and f.terms[0] not in used_terms: 
                    continue
                rule_bodies.append(base + (f,))
        return rule_bodies

    def filter_invalid(rbody):
        # filter invalid rule bodies
        channels = [f.terms[0] for f in rbody if f.rel in ["enc_type", "field_type", "aggregate", "bin", "channel"]]
        for c in channels:
            if channels.count(c) == 1:
                return False
        return True

    rule_bodies = [x for l in [enumerate_rule_body(facts, size) for size in range(1, size + 1) ]for x in l]
    rule_bodies = set([canonicalize(rb) for rb in rule_bodies if filter_invalid(rb)])

    return rule_bodies


def infer_rule_tree(facts, max_size=3):
    """Given properties of a spec, infer relations over the spec"""
    rules = []

    head = Atom("invalid", ["v"])

    rule_bodies = enumerate_rules(facts, max_size)

    # infer abstract rule bodies
    abs_rule_bodies = {}
    for rbody in rule_bodies:
        abs_rule_body = lift_rule_body(rbody)
        equiv_rbodies = gen_alpha_equivalent_rbody(abs_rule_body)

        abs_exists = False
        for rb in equiv_rbodies:
            if rb in abs_rule_bodies:
                abs_rule_bodies[rb].append(rbody)
                abs_exists = True
                break

        if not abs_exists:
            abs_rule_bodies[abs_rule_body] = [rbody]

    return abs_rule_bodies


def eliminate_pos(rels, pos_rels):
    return [x for x in rels if x not in pos_rels]


def union_rule_tree(tr1, tr2):
    tr = tr1 #copy.deepcopy(tr1)
    for rule in tr2:
        rule_in_tr = False
        for alpha_equiv_rule in gen_alpha_equivalent_rbody(rule):
            if alpha_equiv_rule in tr:
                tr[alpha_equiv_rule] = tr[alpha_equiv_rule] + tr2[rule]
                rule_in_tr = True
                break
        if not rule_in_tr:
            tr[rule] = tr2[rule]
    return tr

def substract_rule_tree(tr1, tr2):
    tr = tr1
    kept_concrete_rules = []
    for rule in tr2:
        for alpha_equiv_rule in gen_alpha_equivalent_rbody(rule):
            if alpha_equiv_rule in tr:
                remaining_concrete_rules = []
                concrete_rules = tr.pop(alpha_equiv_rule)
                for r in concrete_rules:
                    if r not in tr2[rule]:
                        remaining_concrete_rules.append(r)
                kept_concrete_rules = kept_concrete_rules + remaining_concrete_rules

    # for r in kept_concrete_rules:
    #     tr[r] = [r]
    return tr

def main():
    synth_data = data_utils.load_synthetic_dataset()
    codar_data = data_utils.load_codar_dataset()

    rtree = {}
    for i, d in enumerate(sorted(synth_data + codar_data, key=lambda x: x["label"])):
        print(f'{i}: {d["label"]}')
        facts = infer_facts(d["vl"], d["data_schema"]) 
        tr = infer_rule_tree(facts)
        if d["label"] == False:
            rtree = union_rule_tree(rtree, tr)
        elif d["label"] == True:
            rtree = substract_rule_tree(rtree, tr)
    
    for rbody in rtree:
        print(Rule(Atom("invalid", ["v"]), rbody))

        # print(d)
        # facts = infer_facts(d["vl"], d["data_schema"]) 
        # rule_bodies = infer_rule_tree(facts)

        # pprint(facts)
        # pprint(rule_bodies)
        # sys.exit(-1)

    #pprint(loaded_data)

if __name__ == '__main__':
    main()

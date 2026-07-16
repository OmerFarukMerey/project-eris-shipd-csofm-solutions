#!/usr/bin/env python3
"""CPU-only supervised solution for Causal Pathway Event Repair.

The primary model is a CatBoost candidate ranker over typed local-graph features.
Participant roles, regulation, ambiguity, and confidence are learned by separate
cross-fitted heads.  Public IDs and candidate positions are never used as model
features.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
SEED = 2026
THREADS = min(10, os.cpu_count() or 1)
ROLES = ["input", "output", "catalyst", "activator", "inhibitor"]
REGS = ["neutral", "activation", "inhibition"]
PROFILE_FIELDS = ("entity_type", "compartment", "state_class", "component_bin")
ROLE_THRESHOLDS = {
    "input": 0.40,
    "output": 0.35,
    "catalyst": 0.35,
    "activator": 0.30,
    "inhibitor": 0.40,
}
ABSTAIN_THRESHOLD = 0.30


def resolve_paths() -> tuple[Path, Path]:
    """Resolve platform-supplied paths, with a local no-argument fallback."""
    here = Path(__file__).resolve().parent
    if len(sys.argv) == 3:
        public_dir = Path(sys.argv[1])
        submission_out = Path(sys.argv[2])
    elif len(sys.argv) == 1:
        candidates = [
            here / "dataset" / "public",
            here / "public",
            Path("dataset/public"),
            Path("public"),
        ]
        public_dir = next(
            (
                path
                for path in candidates
                if (path / "train.csv").exists() and (path / "test.csv").exists()
            ),
            None,
        )
        if public_dir is None:
            raise FileNotFoundError("Could not locate public/train.csv and public/test.csv")
        submission_out = here / "working" / "submission.csv"
    else:
        raise SystemExit("Usage: python3 solution.py <public_dir> <submission_out>")
    if not (public_dir / "train.csv").is_file() or not (public_dir / "test.csv").is_file():
        raise FileNotFoundError(f"Missing train.csv or test.csv under {public_dir}")
    return public_dir, submission_out


def profile(entity: dict, fields=PROFILE_FIELDS) -> tuple:
    return tuple(entity[k] for k in fields)


def profile_token(entity: dict) -> str:
    return "|".join(profile(entity))


def multiset_intersection(a, b) -> int:
    return sum((Counter(a) & Counter(b)).values())


def add_stats(target: dict, prefix: str, values) -> None:
    values = list(values)
    if values:
        target[prefix + "_min"] = min(values)
        target[prefix + "_max"] = max(values)
        target[prefix + "_mean"] = sum(values) / len(values)
        target[prefix + "_sum"] = sum(values)
    else:
        target[prefix + "_min"] = 0
        target[prefix + "_max"] = 0
        target[prefix + "_mean"] = 0.0
        target[prefix + "_sum"] = 0


def categorical_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]


def parse_frame(frame: pd.DataFrame):
    contexts = [json.loads(x) for x in frame.context_json]
    candidates = [json.loads(x) for x in frame.candidates_json]
    return contexts, candidates


def make_folds(train: pd.DataFrame, contexts: list[dict], candidates: list[list[dict]]) -> np.ndarray:
    """Keep rows with substantially overlapping event-card pools in one fold.

    The release isolates related source fragments across train and test. Random
    row folds therefore overestimate generalization. These groups are derived
    only from released coarse descriptors, never IDs or source metadata.
    """
    selected_types = []
    card_rows = defaultdict(list)
    for i, (row, ctx, cards) in enumerate(zip(train.itertuples(), contexts, candidates)):
        if row.abstain:
            selected_types.append("ABSTAIN")
        else:
            selected_types.append(next(c["event_type"] for c in cards if c["event_id"] == row.event_id))
        entities = {e["entity_id"]: e for e in ctx["entities"]}
        signatures = {
            (
                card["event_type"],
                card["compartment"],
                tuple(sorted(profile_token(entities[x]) for x in card["participants"])),
            )
            for card in cards
        }
        for signature in signatures:
            card_rows[signature].append(i)

    shared = Counter()
    for rows in card_rows.values():
        if len(rows) > 50:
            continue
        for ai, left in enumerate(rows):
            for right in rows[ai + 1 :]:
                shared[(left, right)] += 1
    parent = list(range(len(train)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for (left, right), count in shared.items():
        if count >= 4:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root
    groups = [find(i) for i in range(len(train))]

    raw, fallback = [], []
    for i, ctx in enumerate(contexts):
        n_in = sum(b == "MISSING" for a, b in ctx["links"])
        n_out = sum(a == "MISSING" for a, b in ctx["links"])
        raw.append(f"{int(ctx['branching'])}_{n_in}_{n_out}_{selected_types[i]}")
        fallback.append(f"topo_{n_in}_{n_out}_{int(train.abstain.iloc[i])}")
    counts = Counter(raw)
    strata = [x if counts[x] >= 5 else fallback[i] for i, x in enumerate(raw)]
    splitter = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
    folds = np.zeros(len(train), dtype=np.int8)
    for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(train)), strata, groups)):
        folds[valid] = fold
    return folds


def graph_info(ctx: dict, cards: list[dict]):
    event_map = {e["event_id"]: e for e in ctx["events"]}
    entities = {e["entity_id"]: e for e in ctx["entities"]}
    incoming = [event_map[a] for a, b in ctx["links"] if b == "MISSING"]
    outgoing = [event_map[b] for a, b in ctx["links"] if a == "MISSING"]
    predecessors, successors = defaultdict(list), defaultdict(list)
    for a, b in ctx["links"]:
        if a != "MISSING" and b != "MISSING":
            successors[a].append(b)
            predecessors[b].append(a)
    ancestors = [event_map[x] for e in incoming for x in predecessors[e["event_id"]]]
    descendants = [event_map[x] for e in outgoing for x in successors[e["event_id"]]]
    alias_frequency = Counter(x for card in cards for x in card["participants"])
    return event_map, entities, incoming, outgoing, ancestors, descendants, alias_frequency


def candidate_features(ctx: dict, cards: list[dict], ci: int) -> dict:
    card = cards[ci]
    event_map, entities, incoming, outgoing, ancestors, descendants, alias_freq = graph_info(ctx, cards)
    participants = [entities[x] for x in card["participants"]]
    f = {
        "branch": int(ctx["branching"]),
        "n_events": len(ctx["events"]),
        "n_links": len(ctx["links"]),
        "n_entities": len(ctx["entities"]),
        "n_candidates": len(cards),
        "n_in": len(incoming),
        "n_out": len(outgoing),
        "missing_position": f"{len(incoming)}in_{len(outgoing)}out",
        "candidate_type": card["event_type"],
        "candidate_comp": card["compartment"],
        "candidate_n": len(participants),
        "in_types": "+".join(sorted(e["event_type"] for e in incoming)) or "NONE",
        "out_types": "+".join(sorted(e["event_type"] for e in outgoing)) or "NONE",
        "in_comps": "+".join(sorted(e["compartment"] for e in incoming)) or "NONE",
        "out_comps": "+".join(sorted(e["compartment"] for e in outgoing)) or "NONE",
        "in_regs": "+".join(sorted(e["regulation"] for e in incoming)) or "NONE",
        "out_regs": "+".join(sorted(e["regulation"] for e in outgoing)) or "NONE",
    }
    f["candidate_context_types"] = card["event_type"] + ">" + f["in_types"] + ">" + f["out_types"]
    f["candidate_context_comp"] = card["compartment"] + ">" + f["in_comps"] + ">" + f["out_comps"]
    f["profile_signature"] = card["event_type"] + "|" + card["compartment"] + "|" + ",".join(
        sorted(profile_token(p) for p in participants)
    )
    for event_type in EVENT_TYPES:
        f["all_evtype_" + event_type] = sum(e["event_type"] == event_type for e in ctx["events"])
        f["in_evtype_" + event_type] = sum(e["event_type"] == event_type for e in incoming)
        f["out_evtype_" + event_type] = sum(e["event_type"] == event_type for e in outgoing)
        f["candpool_evtype_" + event_type] = sum(c["event_type"] == event_type for c in cards)
    for regulation in REGS:
        f["all_reg_" + regulation] = sum(e["regulation"] == regulation for e in ctx["events"])
        f["in_reg_" + regulation] = sum(e["regulation"] == regulation for e in incoming)
        f["out_reg_" + regulation] = sum(e["regulation"] == regulation for e in outgoing)
    for role in ROLES:
        f["all_role_" + role] = sum(p["role"] == role for e in ctx["events"] for p in e["participants"])
        f["in_role_" + role] = sum(p["role"] == role for e in incoming for p in e["participants"])
        f["out_role_" + role] = sum(p["role"] == role for e in outgoing for p in e["participants"])
    value_groups = [
        ("etype", "entity_type", ENTITY_TYPES),
        ("comp", "compartment", COMPS),
        ("state", "state_class", STATES),
        ("bin", "component_bin", BINS),
    ]
    for prefix, key, values in value_groups:
        for value in values:
            f[f"p_{prefix}_{value}"] = sum(p[key] == value for p in participants)
    f["same_event_comp"] = sum(p["compartment"] == card["compartment"] for p in participants)
    frequencies = [alias_freq[x] for x in card["participants"]]
    add_stats(f, "alias_freq", frequencies)
    f["alias_unique"] = sum(x == 1 for x in frequencies)
    f["alias_shared"] = sum(x > 1 for x in frequencies)
    card_set = set(card["participants"])
    overlaps = [len(card_set & set(c["participants"])) for j, c in enumerate(cards) if j != ci]
    add_stats(f, "other_overlap", overlaps)
    f["other_overlap_positive"] = sum(x > 0 for x in overlaps)
    same_type = [
        len(card_set & set(c["participants"]))
        for j, c in enumerate(cards)
        if j != ci and c["event_type"] == card["event_type"]
    ]
    add_stats(f, "same_type_overlap", same_type)
    groups = {"in": incoming, "out": outgoing, "anc": ancestors, "desc": descendants, "all": ctx["events"]}
    subsets = [
        ("exact", PROFILE_FIELDS),
        ("type_comp", ("entity_type", "compartment")),
        ("type_state", ("entity_type", "state_class")),
        ("type", ("entity_type",)),
        ("comp", ("compartment",)),
        ("state", ("state_class",)),
        ("bin", ("component_bin",)),
    ]
    for group_name, events in groups.items():
        for role in ROLES + ["any"]:
            observed = [p for e in events for p in e["participants"] if role == "any" or p["role"] == role]
            for subset_name, fields in subsets:
                f[f"match_{group_name}_{role}_{subset_name}"] = multiset_intersection(
                    [profile(p, fields) for p in participants], [profile(p, fields) for p in observed]
                )
        event_matches = [
            multiset_intersection([profile(p) for p in participants], [profile(p) for p in e["participants"]])
            for e in events
        ]
        add_stats(f, "eventmatch_" + group_name, event_matches)
    f["count_delta_in"] = len(participants) - sum(len(e["participants"]) for e in incoming)
    f["count_delta_out"] = len(participants) - sum(len(e["participants"]) for e in outgoing)
    same_type_counts = [c["participant_count"] for c in cards if c["event_type"] == card["event_type"]]
    f["count_vs_type_mean"] = len(participants) - sum(same_type_counts) / len(same_type_counts)
    return f


def make_candidate_frame(contexts, candidates, labels=None) -> pd.DataFrame:
    rows = []
    for i, (ctx, cards) in enumerate(zip(contexts, candidates)):
        for ci, card in enumerate(cards):
            f = candidate_features(ctx, cards, ci)
            f["row"] = i
            f["ci"] = ci
            if labels is not None:
                f["target"] = int(labels.abstain.iloc[i] == 0 and labels.event_id.iloc[i] == card["event_id"])
            rows.append(f)
    return pd.DataFrame(rows)


def intrinsic_features(event_type: str, event_comp: str, participants: list[dict], self_idx: int) -> dict:
    entity = participants[self_idx]
    f = {
        "event_type": event_type,
        "event_comp": event_comp,
        "participant_count": len(participants),
        "self_type": entity["entity_type"],
        "self_comp": entity["compartment"],
        "self_state": entity["state_class"],
        "self_bin": entity["component_bin"],
        "type_event": event_type + "|" + entity["entity_type"],
        "profile_event": event_type + "|" + profile_token(entity),
        "comp_relation": event_comp + "|" + entity["compartment"],
        "self_matches_event_comp": int(event_comp == entity["compartment"]),
        "profile_signature": event_type + "|" + event_comp + "|" + ",".join(
            sorted(profile_token(p) for p in participants)
        ),
    }
    value_groups = [
        ("type", "entity_type", ENTITY_TYPES),
        ("comp", "compartment", COMPS),
        ("state", "state_class", STATES),
        ("bin", "component_bin", BINS),
    ]
    for prefix, key, values in value_groups:
        for value in values:
            f[f"count_{prefix}_{value}"] = sum(p[key] == value for p in participants)
        f["same_" + prefix] = sum(p[key] == entity[key] for p in participants)
    f["same_profile"] = sum(profile(p) == profile(entity) for p in participants)
    return f


def make_intrinsic_training(contexts, candidates, labels) -> pd.DataFrame:
    rows = []
    for i, (ctx, cards) in enumerate(zip(contexts, candidates)):
        for event in ctx["events"]:
            for pi, participant in enumerate(event["participants"]):
                f = intrinsic_features(event["event_type"], event["compartment"], event["participants"], pi)
                f.update(row=i, source="known", alias=f"K{pi}")
                for role in ROLES:
                    f["y_" + role] = int(participant["role"] == role)
                rows.append(f)
        if labels.abstain.iloc[i] == 0:
            ci = next(j for j, c in enumerate(cards) if c["event_id"] == labels.event_id.iloc[i])
            card = cards[ci]
            entities = {e["entity_id"]: e for e in ctx["entities"]}
            participants = [entities[x] for x in card["participants"]]
            truth = set(map(tuple, json.loads(labels.participant_roles_json.iloc[i])))
            for pi, alias in enumerate(card["participants"]):
                f = intrinsic_features(card["event_type"], card["compartment"], participants, pi)
                f.update(row=i, source="candidate", alias=alias)
                for role in ROLES:
                    f["y_" + role] = int((alias, role) in truth)
                rows.append(f)
    return pd.DataFrame(rows)


def make_all_intrinsic(contexts, candidates) -> pd.DataFrame:
    rows = []
    for i, (ctx, cards) in enumerate(zip(contexts, candidates)):
        entities = {e["entity_id"]: e for e in ctx["entities"]}
        for ci, card in enumerate(cards):
            participants = [entities[x] for x in card["participants"]]
            for pi, alias in enumerate(card["participants"]):
                f = intrinsic_features(card["event_type"], card["compartment"], participants, pi)
                f.update(row=i, ci=ci, alias=alias)
                rows.append(f)
    return pd.DataFrame(rows)


def add_multitask_candidate_features(cf, contexts, candidates, intrinsic) -> list[str]:
    lookup = {(int(z.row), int(z.ci)): idx for idx, z in cf[["row", "ci"]].iterrows()}
    extra = []
    for role in ROLES:
        for stat in ("sum", "max", "mean"):
            col = f"intr_{role}_{stat}"
            cf[col] = 0.0
            extra.append(col)
    expected_cols = [
        "expected_forward_exact",
        "expected_reverse_exact",
        "expected_forward_typecomp",
        "expected_forward_type",
        "expected_regulator_match",
    ]
    for col in expected_cols:
        cf[col] = 0.0
    extra.extend(expected_cols)
    for (i, ci), group in intrinsic.groupby(["row", "ci"], sort=False):
        i, ci = int(i), int(ci)
        idx = lookup[(i, ci)]
        ctx = contexts[i]
        event_map = {e["event_id"]: e for e in ctx["events"]}
        entities = {e["entity_id"]: e for e in ctx["entities"]}
        incoming = [event_map[a] for a, b in ctx["links"] if b == "MISSING"]
        outgoing = [event_map[b] for a, b in ctx["links"] if a == "MISSING"]
        in_output = [p for e in incoming for p in e["participants"] if p["role"] == "output"]
        in_input = [p for e in incoming for p in e["participants"] if p["role"] == "input"]
        out_input = [p for e in outgoing for p in e["participants"] if p["role"] == "input"]
        out_output = [p for e in outgoing for p in e["participants"] if p["role"] == "output"]
        forward = reverse = forward_tc = forward_type = regulator = 0.0
        for _, z in group.iterrows():
            entity = entities[z.alias]
            m_in = sum(profile(entity) == profile(p) for p in in_output)
            m_out = sum(profile(entity) == profile(p) for p in out_input)
            m_in_reverse = sum(profile(entity) == profile(p) for p in in_input)
            m_out_reverse = sum(profile(entity) == profile(p) for p in out_output)
            tc = ("entity_type", "compartment")
            m_in_tc = sum(profile(entity, tc) == profile(p, tc) for p in in_output)
            m_out_tc = sum(profile(entity, tc) == profile(p, tc) for p in out_input)
            m_in_type = sum(entity["entity_type"] == p["entity_type"] for p in in_output)
            m_out_type = sum(entity["entity_type"] == p["entity_type"] for p in out_input)
            forward += z.p_input * m_in + z.p_output * m_out
            reverse += z.p_output * m_in_reverse + z.p_input * m_out_reverse
            forward_tc += z.p_input * m_in_tc + z.p_output * m_out_tc
            forward_type += z.p_input * m_in_type + z.p_output * m_out_type
            regulator += (z.p_activator + z.p_inhibitor) * (m_in + m_out)
        for role in ROLES:
            cf.loc[idx, f"intr_{role}_sum"] = group["p_" + role].sum()
            cf.loc[idx, f"intr_{role}_max"] = group["p_" + role].max()
            cf.loc[idx, f"intr_{role}_mean"] = group["p_" + role].mean()
        cf.loc[idx, expected_cols] = [forward, reverse, forward_tc, forward_type, regulator]
    network_cols = [
        "card_degree",
        "card_weighted_degree",
        "card_triangle",
        "card_component_size",
        "card_overlap_type_same",
        "card_overlap_comp_same",
    ] + ["card_neighbor_type_" + t for t in EVENT_TYPES]
    for col in network_cols:
        cf[col] = 0.0
    extra.extend(network_cols)
    for i, cards in enumerate(candidates):
        sets = [set(c["participants"]) for c in cards]
        adjacency = np.array(
            [[len(sets[a] & sets[b]) if a != b else 0 for b in range(len(cards))] for a in range(len(cards))]
        )
        binary = adjacency > 0
        seen, component_size = set(), {}
        for start in range(len(cards)):
            if start in seen:
                continue
            stack, component = [start], []
            seen.add(start)
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbor in np.where(binary[node])[0]:
                    neighbor = int(neighbor)
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            for node in component:
                component_size[node] = len(component)
        for ci, card in enumerate(cards):
            idx = lookup[(i, ci)]
            neighbors = list(np.where(binary[ci])[0])
            cf.loc[idx, "card_degree"] = len(neighbors)
            cf.loc[idx, "card_weighted_degree"] = adjacency[ci].sum()
            cf.loc[idx, "card_component_size"] = component_size[ci]
            cf.loc[idx, "card_overlap_type_same"] = sum(
                adjacency[ci, j] for j in neighbors if cards[j]["event_type"] == card["event_type"]
            )
            cf.loc[idx, "card_overlap_comp_same"] = sum(
                adjacency[ci, j] for j in neighbors if cards[j]["compartment"] == card["compartment"]
            )
            cf.loc[idx, "card_triangle"] = sum(
                binary[a, b] for ai, a in enumerate(neighbors) for b in neighbors[ai + 1 :]
            )
            for event_type in EVENT_TYPES:
                cf.loc[idx, "card_neighbor_type_" + event_type] = sum(
                    adjacency[ci, j] for j in neighbors if cards[j]["event_type"] == event_type
                )
    return extra


def participant_features(ctx: dict, cards: list[dict], ci: int, alias: str) -> dict:
    f = candidate_features(ctx, cards, ci)
    card = cards[ci]
    event_map, entities, incoming, outgoing, ancestors, descendants, alias_freq = graph_info(ctx, cards)
    entity = entities[alias]
    participants = [entities[x] for x in card["participants"]]
    f.update(
        entity_type_self=entity["entity_type"],
        entity_comp_self=entity["compartment"],
        entity_state_self=entity["state_class"],
        entity_bin_self=entity["component_bin"],
        entity_event_type=card["event_type"] + "|" + profile_token(entity),
        entity_event_comp=card["event_type"] + "|" + card["compartment"] + "|" + entity["compartment"],
        self_alias_freq=alias_freq[alias],
        self_same_candidate_profile=sum(profile(x) == profile(entity) for x in participants),
        self_same_context_profile=sum(profile(x) == profile(entity) for x in entities.values()),
        self_same_candidate_type=sum(x["entity_type"] == entity["entity_type"] for x in participants),
        self_same_candidate_comp=sum(x["compartment"] == entity["compartment"] for x in participants),
    )
    containing = [c for c in cards if alias in c["participants"]]
    f["self_card_types"] = "+".join(sorted(c["event_type"] for c in containing))
    f["self_card_comps"] = "+".join(sorted(c["compartment"] for c in containing))
    f["self_same_type_cards"] = sum(c["event_type"] == card["event_type"] for c in containing)
    groups = {"in": incoming, "out": outgoing, "anc": ancestors, "desc": descendants, "all": ctx["events"]}
    subsets = [
        ("exact", PROFILE_FIELDS),
        ("type_comp", ("entity_type", "compartment")),
        ("type_state", ("entity_type", "state_class")),
        ("type", ("entity_type",)),
        ("comp", ("compartment",)),
        ("state", ("state_class",)),
        ("bin", ("component_bin",)),
    ]
    for group_name, events in groups.items():
        for role in ROLES + ["any"]:
            observed = [p for e in events for p in e["participants"] if role == "any" or p["role"] == role]
            for subset_name, fields in subsets:
                f[f"self_match_{group_name}_{role}_{subset_name}"] = sum(
                    profile(entity, fields) == profile(p, fields) for p in observed
                )
    return f


def make_true_role_frame(contexts, candidates, labels) -> pd.DataFrame:
    rows = []
    for i, (ctx, cards) in enumerate(zip(contexts, candidates)):
        if labels.abstain.iloc[i]:
            continue
        ci = next(j for j, c in enumerate(cards) if c["event_id"] == labels.event_id.iloc[i])
        truth = set(map(tuple, json.loads(labels.participant_roles_json.iloc[i])))
        for alias in cards[ci]["participants"]:
            f = participant_features(ctx, cards, ci, alias)
            f.update(row=i, ci=ci, alias=alias)
            for role in ROLES:
                f["y_" + role] = int((alias, role) in truth)
            rows.append(f)
    return pd.DataFrame(rows)


def make_chosen_role_frame(contexts, candidates, chosen) -> pd.DataFrame:
    rows = []
    for i, (ctx, cards) in enumerate(zip(contexts, candidates)):
        ci = int(chosen[i])
        for alias in cards[ci]["participants"]:
            f = participant_features(ctx, cards, ci, alias)
            f.update(row=i, ci=ci, alias=alias)
            rows.append(f)
    return pd.DataFrame(rows)


def ambiguity_base_features(ctx: dict, cards: list[dict]) -> dict:
    event_map = {e["event_id"]: e for e in ctx["events"]}
    pred, succ = Counter(), Counter()
    for a, b in ctx["links"]:
        succ[a] += 1
        pred[b] += 1
    incoming = [a for a, b in ctx["links"] if b == "MISSING"]
    outgoing = [b for a, b in ctx["links"] if a == "MISSING"]
    f = {
        "branch": int(ctx["branching"]),
        "n_in": len(incoming),
        "n_out": len(outgoing),
        "n_events": len(ctx["events"]),
        "n_links": len(ctx["links"]),
        "n_cands": len(cards),
        "n_entities": len(ctx["entities"]),
        "position": f"{len(incoming)}in_{len(outgoing)}out",
        "missing_end": int(not outgoing),
        "missing_start": int(not incoming),
        "inc_other_out_sum": sum(max(0, succ[x] - 1) for x in incoming),
        "inc_out_degree_max": max([succ[x] for x in incoming] or [0]),
        "inc_in_degree_sum": sum(pred[x] for x in incoming),
        "out_other_in_sum": sum(max(0, pred[x] - 1) for x in outgoing),
        "out_in_degree_max": max([pred[x] for x in outgoing] or [0]),
        "out_out_degree_sum": sum(succ[x] for x in outgoing),
        "known_split_nodes": sum(succ[x] > 1 for x in event_map),
        "known_merge_nodes": sum(pred[x] > 1 for x in event_map),
        "inc_types": "+".join(sorted(event_map[x]["event_type"] for x in incoming)) or "NONE",
        "out_types": "+".join(sorted(event_map[x]["event_type"] for x in outgoing)) or "NONE",
        "inc_comps": "+".join(sorted(event_map[x]["compartment"] for x in incoming)) or "NONE",
        "out_comps": "+".join(sorted(event_map[x]["compartment"] for x in outgoing)) or "NONE",
        "inc_regs": "+".join(sorted(event_map[x]["regulation"] for x in incoming)) or "NONE",
        "out_regs": "+".join(sorted(event_map[x]["regulation"] for x in outgoing)) or "NONE",
    }
    for event_type in EVENT_TYPES:
        f["all_type_" + event_type] = sum(e["event_type"] == event_type for e in ctx["events"])
        f["cand_type_" + event_type] = sum(c["event_type"] == event_type for c in cards)
    return f


def add_score_distribution_features(amb: pd.DataFrame, candidate_frame: pd.DataFrame) -> None:
    for i, group in candidate_frame.groupby("row", sort=False):
        for prefix, col in (("base", "base_score"), ("stage2", "stage2_score"), ("blend", "blend_score")):
            scores = np.sort(group[col].to_numpy())[::-1]
            amb.loc[i, prefix + "_top"] = scores[0]
            amb.loc[i, prefix + "_second"] = scores[1]
            amb.loc[i, prefix + "_gap"] = scores[0] - scores[1]
            amb.loc[i, prefix + "_mean"] = scores.mean()
            amb.loc[i, prefix + "_std"] = scores.std()
            amb.loc[i, prefix + "_top_ratio"] = scores[0] / (abs(scores[1]) + 1e-6)
        amb.loc[i, "stage_agree"] = int(group.base_score.idxmax() == group.stage2_score.idxmax())


def decode_roles(group: pd.DataFrame) -> set[tuple[str, str]]:
    pairs = set()
    for _, z in group.iterrows():
        probs = np.array([z["p_" + role] for role in ROLES])
        chosen = [role for role, p in zip(ROLES, probs) if p >= ROLE_THRESHOLDS[role]]
        if not chosen:
            chosen = [ROLES[int(np.argmax(probs))]]
        pairs.update((z.alias, role) for role in chosen)
    return pairs


def attach_role_aggregates(frame: pd.DataFrame, role_frame: pd.DataFrame) -> None:
    for role in ROLES:
        grouped = role_frame.groupby("row")["p_" + role].agg(["max", "mean", "sum"])
        for stat in ("max", "mean", "sum"):
            frame[f"rolep_{role}_{stat}"] = frame.row.map(grouped[stat])
        counts = (role_frame["p_" + role] >= ROLE_THRESHOLDS[role]).groupby(role_frame.row).sum()
        frame[f"rolep_{role}_n"] = frame.row.map(counts)


def selected_candidate_frame(cf: pd.DataFrame, chosen: list[int]) -> pd.DataFrame:
    rows = []
    for i, ci in enumerate(chosen):
        rows.append(cf[(cf.row == i) & (cf.ci == ci)].iloc[0].to_dict())
    return pd.DataFrame(rows).reset_index(drop=True)


def evaluate_outcomes(train, candidates, chosen, abstain_probability, role_sets, reg_pred):
    records = []
    true_roles = [set(map(tuple, json.loads(x))) for x in train.participant_roles_json]
    for i in range(len(train)):
        do_abstain = abstain_probability[i] >= ABSTAIN_THRESHOLD
        pred_event = "ABSTAIN" if do_abstain else candidates[i][chosen[i]]["event_id"]
        pred_roles = set() if do_abstain else role_sets[i]
        pred_reg = "neutral" if do_abstain else reg_pred[i]
        pred_abstain = int(do_abstain)
        event_score = int(pred_event == train.event_id.iloc[i])
        denom = len(pred_roles) + len(true_roles[i])
        role_f1 = 1.0 if denom == 0 else 2 * len(pred_roles & true_roles[i]) / denom
        reg_score = int(pred_reg == train.regulation.iloc[i])
        abstain_score = int(pred_abstain == train.abstain.iloc[i])
        repair = event_score * role_f1
        points = 0.54 * repair + 0.14 * reg_score + 0.18 * abstain_score
        correctness = points / 0.86
        consistency = int(
            event_score and pred_roles == true_roles[i] and reg_score and abstain_score
        )
        records.append((event_score, role_f1, reg_score, abstain_score, repair, points, correctness, consistency))
    return pd.DataFrame(
        records,
        columns=["event", "role_f1", "reg", "abstain_ok", "repair", "points", "correctness", "consistency"],
    )


def confidence_features(
    amb: pd.DataFrame,
    amb_cols: list[str],
    candidate_frame: pd.DataFrame,
    chosen: list[int],
    role_frame: pd.DataFrame,
    regulation: list[str],
    reg_confidence: np.ndarray,
    abstain_probability: np.ndarray,
) -> pd.DataFrame:
    meta = amb[amb_cols].copy()
    meta["abstain_probability"] = abstain_probability
    meta["decision_abstain"] = (abstain_probability >= ABSTAIN_THRESHOLD).astype(int)
    meta["abstain_margin"] = np.abs(abstain_probability - ABSTAIN_THRESHOLD)
    for i, group in candidate_frame.groupby("row", sort=False):
        ci = chosen[int(i)]
        selected = group[group.ci == ci].iloc[0]
        for prefix, col in (("base", "base_score"), ("stage2", "stage2_score")):
            scores = np.sort(group[col].to_numpy())[::-1]
            meta.loc[i, prefix + "_sel_score"] = selected[col]
            meta.loc[i, prefix + "_top_gap"] = scores[0] - scores[1]
        meta.loc[i, "selected_type"] = selected.candidate_type
        meta.loc[i, "selected_comp"] = selected.candidate_comp
        meta.loc[i, "selected_n"] = selected.candidate_n
    meta["reg_pred"] = regulation
    meta["reg_conf"] = reg_confidence
    for i, group in role_frame.groupby("row", sort=False):
        probs = group[["p_" + r for r in ROLES]].to_numpy()
        ordered = np.sort(probs, axis=1)
        meta.loc[i, "role_top_mean"] = ordered[:, -1].mean()
        meta.loc[i, "role_margin_mean"] = (ordered[:, -1] - ordered[:, -2]).mean()
        clipped = np.clip(probs, 1e-6, 1.0)
        meta.loc[i, "role_entropy_mean"] = np.mean(-np.sum(clipped * np.log(clipped), axis=1))
        meta.loc[i, "role_pair_count"] = len(decode_roles(group))
        meta.loc[i, "role_participant_count"] = len(group)
    return meta


def main() -> None:
    started = time.time()
    data_dir, submission_out = resolve_paths()
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    tr_ctx, tr_cards = parse_frame(train)
    te_ctx, te_cards = parse_frame(test)

    global EVENT_TYPES, COMPS, ENTITY_TYPES, STATES, BINS
    # Define the feature vocabulary from public training inputs only. Test rows
    # are transformed with that fixed vocabulary and are never used to fit or
    # configure any model component.
    EVENT_TYPES = sorted(
        {e["event_type"] for ctx in tr_ctx for e in ctx["events"]}
        | {c["event_type"] for cards in tr_cards for c in cards}
    )
    COMPS = sorted({e["compartment"] for ctx in tr_ctx for e in ctx["entities"]})
    ENTITY_TYPES = sorted({e["entity_type"] for ctx in tr_ctx for e in ctx["entities"]})
    STATES = sorted({e["state_class"] for ctx in tr_ctx for e in ctx["entities"]})
    BINS = sorted({e["component_bin"] for ctx in tr_ctx for e in ctx["entities"]})
    folds = make_folds(train, tr_ctx, tr_cards)

    print("[1/7] Extracting graph/candidate features", flush=True)
    cf_train = make_candidate_frame(tr_ctx, tr_cards, train)
    cf_test = make_candidate_frame(te_ctx, te_cards)
    memorization_features = {"profile_signature", "candidate_context_comp", "candidate_context_types"}
    base_features = [
        c for c in cf_train
        if c not in {"row", "ci", "target"} | memorization_features
    ]
    base_cat = categorical_columns(cf_train, base_features)

    print("[2/7] Cross-fitting base candidate ranker", flush=True)
    cf_train["base_score"] = np.nan
    cf_test["base_score"] = 0.0
    for fold in range(4):
        train_rows = np.where(folds != fold)[0]
        valid_rows = np.where(folds == fold)[0]
        train_mask = cf_train.row.isin(train_rows)
        valid_mask = cf_train.row.isin(valid_rows)
        model = CatBoostClassifier(
            iterations=750, depth=6, learning_rate=0.055, loss_function="Logloss",
            cat_features=base_cat, verbose=False, random_seed=100 + fold,
            thread_count=THREADS, l2_leaf_reg=10, random_strength=0.7,
            auto_class_weights="Balanced",
        )
        model.fit(cf_train.loc[train_mask, base_features], cf_train.loc[train_mask, "target"])
        cf_train.loc[valid_mask, "base_score"] = model.predict_proba(cf_train.loc[valid_mask, base_features])[:, 1]
        cf_test["base_score"] += model.predict_proba(cf_test[base_features])[:, 1] / 4

    print("[3/7] Learning intrinsic participant roles and graph continuity", flush=True)
    intrinsic_train = make_intrinsic_training(tr_ctx, tr_cards, train)
    all_intr_train = make_all_intrinsic(tr_ctx, tr_cards)
    all_intr_test = make_all_intrinsic(te_ctx, te_cards)
    intrinsic_cols = [
        c for c in intrinsic_train
        if c not in {"row", "source", "alias", "profile_signature"} and not c.startswith("y_")
    ]
    intrinsic_cat = categorical_columns(intrinsic_train, intrinsic_cols)
    for role in ROLES:
        all_intr_train["p_" + role] = np.nan
        all_intr_test["p_" + role] = 0.0
    intrinsic_models = []
    for fold in range(4):
        fit_mask = intrinsic_train.row.map(lambda x: folds[int(x)] != fold)
        predict_mask = all_intr_train.row.map(lambda x: folds[int(x)] == fold)
        model = CatBoostClassifier(
            iterations=400, depth=7, learning_rate=0.075, loss_function="MultiLogloss",
            cat_features=intrinsic_cat, verbose=False, thread_count=THREADS,
            random_seed=500 + fold, l2_leaf_reg=8, random_strength=0.7,
        )
        model.fit(
            intrinsic_train.loc[fit_mask, intrinsic_cols],
            intrinsic_train.loc[fit_mask, ["y_" + r for r in ROLES]],
        )
        pred = np.asarray(model.predict_proba(all_intr_train.loc[predict_mask, intrinsic_cols]))
        test_pred = np.asarray(model.predict_proba(all_intr_test[intrinsic_cols]))
        for j, role in enumerate(ROLES):
            all_intr_train.loc[predict_mask, "p_" + role] = pred[:, j]
            all_intr_test["p_" + role] += test_pred[:, j] / 4
        intrinsic_models.append(model)
    extra_train = add_multitask_candidate_features(cf_train, tr_ctx, tr_cards, all_intr_train)
    extra_test = add_multitask_candidate_features(cf_test, te_ctx, te_cards, all_intr_test)
    assert extra_train == extra_test
    cf_train["base_candidate_score"] = cf_train.base_score
    cf_test["base_candidate_score"] = cf_test.base_score
    stage_features = base_features + extra_train + ["base_candidate_score"]
    stage_cat = categorical_columns(cf_train, stage_features)

    print("[4/7] Cross-fitting multitask candidate and ambiguity heads", flush=True)
    cf_train["stage2_score"] = np.nan
    cf_test["stage2_score"] = 0.0
    for fold in range(4):
        train_rows = np.where(folds != fold)[0]
        valid_rows = np.where(folds == fold)[0]
        train_mask = cf_train.row.isin(train_rows)
        valid_mask = cf_train.row.isin(valid_rows)
        model = CatBoostClassifier(
            iterations=800, depth=7, learning_rate=0.052, loss_function="Logloss",
            cat_features=stage_cat, verbose=False, random_seed=1100 + fold,
            thread_count=THREADS, l2_leaf_reg=10, random_strength=0.7,
            auto_class_weights="Balanced",
        )
        model.fit(cf_train.loc[train_mask, stage_features], cf_train.loc[train_mask, "target"])
        cf_train.loc[valid_mask, "stage2_score"] = model.predict_proba(cf_train.loc[valid_mask, stage_features])[:, 1]
        cf_test["stage2_score"] += model.predict_proba(cf_test[stage_features])[:, 1] / 4
    for frame in (cf_train, cf_test):
        # The role-aware stage is materially stronger under pathway-isolated
        # validation; the base score remains an input feature and diagnostic.
        frame["blend_score"] = frame["stage2_score"]
    chosen_train = [
        int(group.loc[group.blend_score.idxmax(), "ci"])
        for _, group in cf_train.groupby("row", sort=False)
    ]
    chosen_test = [
        int(group.loc[group.blend_score.idxmax(), "ci"])
        for _, group in cf_test.groupby("row", sort=False)
    ]
    amb_train = pd.DataFrame([ambiguity_base_features(c, cs) for c, cs in zip(tr_ctx, tr_cards)])
    amb_test = pd.DataFrame([ambiguity_base_features(c, cs) for c, cs in zip(te_ctx, te_cards)])
    add_score_distribution_features(amb_train, cf_train)
    add_score_distribution_features(amb_test, cf_test)
    amb_features = list(amb_train.columns)
    amb_cat = categorical_columns(amb_train, amb_features)
    amb_train["abstain_probability"] = np.nan
    amb_test["abstain_probability"] = 0.0
    for fold in range(4):
        fit_mask = folds != fold
        valid_mask = folds == fold
        model = CatBoostClassifier(
            iterations=550, depth=6, learning_rate=0.055, loss_function="Logloss",
            cat_features=amb_cat, verbose=False, thread_count=THREADS,
            random_seed=1200 + fold, l2_leaf_reg=8, random_strength=0.6,
            auto_class_weights="SqrtBalanced",
        )
        model.fit(amb_train.loc[fit_mask, amb_features], train.abstain.loc[fit_mask])
        amb_train.loc[valid_mask, "abstain_probability"] = model.predict_proba(
            amb_train.loc[valid_mask, amb_features]
        )[:, 1]
        amb_test["abstain_probability"] += model.predict_proba(amb_test[amb_features])[:, 1] / 4

    print("[5/7] Cross-fitting contextual role and regulation heads", flush=True)
    true_roles = make_true_role_frame(tr_ctx, tr_cards, train)
    chosen_roles_train = make_chosen_role_frame(tr_ctx, tr_cards, chosen_train)
    chosen_roles_test = make_chosen_role_frame(te_ctx, te_cards, chosen_test)
    role_features = [
        c for c in true_roles
        if c not in {"row", "ci", "alias"} | memorization_features and not c.startswith("y_")
    ]
    role_cat = categorical_columns(true_roles, role_features)
    for frame in (true_roles, chosen_roles_train, chosen_roles_test):
        for role in ROLES:
            frame["p_context_" + role] = 0.0 if frame is chosen_roles_test else np.nan
    for fold in range(4):
        fit_mask = true_roles.row.map(lambda x: folds[int(x)] != fold)
        true_valid = true_roles.row.map(lambda x: folds[int(x)] == fold)
        chosen_valid = chosen_roles_train.row.map(lambda x: folds[int(x)] == fold)
        model = CatBoostClassifier(
            iterations=450, depth=7, learning_rate=0.065, loss_function="MultiLogloss",
            cat_features=role_cat, verbose=False, thread_count=THREADS,
            random_seed=300 + fold, l2_leaf_reg=8, random_strength=0.7,
        )
        model.fit(
            true_roles.loc[fit_mask, role_features],
            true_roles.loc[fit_mask, ["y_" + r for r in ROLES]],
        )
        for frame, mask, divisor in (
            (true_roles, true_valid, 1),
            (chosen_roles_train, chosen_valid, 1),
            (chosen_roles_test, np.ones(len(chosen_roles_test), dtype=bool), 4),
        ):
            pred = np.asarray(model.predict_proba(frame.loc[mask, role_features]))
            for j, role in enumerate(ROLES):
                if divisor == 1:
                    frame.loc[mask, "p_context_" + role] = pred[:, j]
                else:
                    frame.loc[mask, "p_context_" + role] += pred[:, j] / divisor

    def merge_intrinsic(role_frame, intrinsic_frame):
        lookup = intrinsic_frame.set_index(["row", "ci", "alias"])[["p_" + r for r in ROLES]]
        for role in ROLES:
            role_frame["p_intrinsic_" + role] = [
                lookup.loc[(int(z.row), int(z.ci), z.alias), "p_" + role]
                for _, z in role_frame[["row", "ci", "alias"]].iterrows()
            ]
            role_frame["p_" + role] = (
                0.75 * role_frame["p_context_" + role] + 0.25 * role_frame["p_intrinsic_" + role]
            )

    merge_intrinsic(true_roles, all_intr_train)
    merge_intrinsic(chosen_roles_train, all_intr_train)
    merge_intrinsic(chosen_roles_test, all_intr_test)
    role_sets_train = {int(i): decode_roles(group) for i, group in chosen_roles_train.groupby("row")}
    role_sets_test = {int(i): decode_roles(group) for i, group in chosen_roles_test.groupby("row")}

    true_selected = cf_train[cf_train.target == 1].copy().reset_index(drop=True)
    attach_role_aggregates(true_selected, true_roles)
    chosen_cf_train = selected_candidate_frame(cf_train, chosen_train)
    chosen_cf_test = selected_candidate_frame(cf_test, chosen_test)
    attach_role_aggregates(chosen_cf_train, chosen_roles_train)
    attach_role_aggregates(chosen_cf_test, chosen_roles_test)
    reg_role_cols = [c for c in true_selected if c.startswith("rolep_")]
    reg_features = base_features + reg_role_cols
    reg_cat = categorical_columns(true_selected, reg_features)
    reg_train_pred = np.empty(len(train), dtype=object)
    reg_train_conf = np.zeros(len(train))
    reg_test_prob = {r: np.zeros(len(test)) for r in REGS}
    for fold in range(4):
        fit_mask = true_selected.row.map(lambda x: folds[int(x)] != fold)
        chosen_valid = chosen_cf_train.row.map(lambda x: folds[int(x)] == fold)
        target = true_selected.row.map(train.regulation)
        model = CatBoostClassifier(
            iterations=600, depth=7, learning_rate=0.06, loss_function="MultiClass",
            cat_features=reg_cat, verbose=False, thread_count=THREADS,
            random_seed=700 + fold, l2_leaf_reg=8, random_strength=0.7,
            class_weights=[1.0, 1.25, 1.5],
        )
        model.fit(true_selected.loc[fit_mask, reg_features], target.loc[fit_mask])
        pred = model.predict_proba(chosen_cf_train.loc[chosen_valid, reg_features])
        rows = chosen_cf_train.loc[chosen_valid, "row"].astype(int).to_numpy()
        reg_train_pred[rows] = model.classes_[np.argmax(pred, axis=1)]
        reg_train_conf[rows] = pred.max(axis=1)
        test_pred = model.predict_proba(chosen_cf_test[reg_features])
        for j, label in enumerate(model.classes_):
            reg_test_prob[label] += test_pred[:, j] / 4
    reg_test_matrix = np.column_stack([reg_test_prob[r] for r in REGS])
    reg_test_pred = np.array(REGS, dtype=object)[np.argmax(reg_test_matrix, axis=1)]
    reg_test_conf = reg_test_matrix.max(axis=1)

    print("[6/7] Calibrating normalized task correctness", flush=True)
    outcomes = evaluate_outcomes(
        train, tr_cards, chosen_train, amb_train.abstain_probability.to_numpy(),
        role_sets_train, reg_train_pred,
    )
    meta_train = confidence_features(
        amb_train, amb_features, cf_train, chosen_train, chosen_roles_train,
        list(reg_train_pred), reg_train_conf, amb_train.abstain_probability.to_numpy(),
    )
    meta_test = confidence_features(
        amb_test, amb_features, cf_test, chosen_test, chosen_roles_test,
        list(reg_test_pred), reg_test_conf, amb_test.abstain_probability.to_numpy(),
    )
    meta_features = list(meta_train.columns)
    meta_cat = categorical_columns(meta_train, meta_features)
    confidence_oof = np.zeros(len(train))
    confidence_test = np.zeros(len(test))
    for fold in range(4):
        fit_mask = folds != fold
        valid_mask = folds == fold
        model = CatBoostRegressor(
            iterations=600, depth=6, learning_rate=0.045, loss_function="MAE",
            cat_features=meta_cat, verbose=False, thread_count=THREADS,
            random_seed=1400 + fold, l2_leaf_reg=10, random_strength=0.8,
        )
        model.fit(meta_train.loc[fit_mask, meta_features], outcomes.correctness.loc[fit_mask])
        confidence_oof[valid_mask] = model.predict(meta_train.loc[valid_mask, meta_features])
        confidence_test += model.predict(meta_test[meta_features]) / 4
    confidence_oof = np.clip(confidence_oof, 0, 1)
    confidence_test = np.clip(confidence_test, 0, 1)
    cv_score = np.mean(
        outcomes.points
        + 0.08 * (1 - np.abs(confidence_oof - outcomes.correctness))
        + 0.06 * outcomes.consistency
    )
    non_abstain = train.abstain.to_numpy() == 0
    correct_candidate = np.array([
        tr_cards[i][chosen_train[i]]["event_id"] == train.event_id.iloc[i]
        for i in range(len(train))
    ])
    print(
        f"OOF candidate accuracy={correct_candidate[non_abstain].mean():.4f}; "
        f"abstention accuracy={outcomes.abstain_ok.mean():.4f}; "
        f"estimated structured score={cv_score:.4f}",
        flush=True,
    )

    print("[7/7] Writing schema-valid submission", flush=True)
    output_rows = []
    for i in range(len(test)):
        if amb_test.abstain_probability.iloc[i] >= ABSTAIN_THRESHOLD:
            event_id = "ABSTAIN"
            pairs = []
            regulation = "neutral"
            abstain = 1
        else:
            event_id = te_cards[i][chosen_test[i]]["event_id"]
            pairs = sorted(role_sets_test[i], key=lambda x: (x[0], ROLES.index(x[1])))
            regulation = str(reg_test_pred[i])
            abstain = 0
        output_rows.append(
            {
                "id": test.id.iloc[i],
                "event_id": event_id,
                "participant_roles_json": json.dumps([list(x) for x in pairs], separators=(",", ":")),
                "regulation": regulation,
                "abstain": abstain,
                "confidence": float(confidence_test[i]),
            }
        )
    submission = pd.DataFrame(output_rows)[
        ["id", "event_id", "participant_roles_json", "regulation", "abstain", "confidence"]
    ]
    submission.to_csv(submission_out, index=False)
    print(
        f"Wrote {submission_out} ({len(submission)} rows) "
        f"in {(time.time() - started) / 60:.1f} minutes",
        flush=True,
    )


if __name__ == "__main__":
    main()

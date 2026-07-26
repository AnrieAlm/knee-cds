# outcome_trigger.py
#
# Deterministic trigger for outcome-measure suggestions.
#
# This lives in backend/safety/ alongside ottawa.py, pittsburgh.py, and
# red_flags.py because it is architecturally the same kind of thing: a
# context-triggered rule with NO LLM involvement. Specific clinical
# context in -> specific, predictable output out. The LLM never decides
# whether an outcome measure is suggested; it only phrases the suggestion
# after this function has decided.
#
# It reads the accumulated case_state, not the user's query. This is what
# keeps outcome measures out of the acute-assessment path: during acute
# triage the case has not progressed, so nothing fires.

from backend.rag.outcome_measures import OUTCOME_MEASURES


# -----------------------------------------------------------
# Case-state -> trigger-tag mapping
# -----------------------------------------------------------
#
# Each rule below inspects one aspect of case_state and, if it applies,
# adds trigger tags. Negation is handled explicitly: a pathology that has
# been RULED OUT must never fire its tags. This is the single most
# important correctness point -- substring matching on an LLM answer that
# says "not an ACL tear" would otherwise fire ACL suggestions.

def _collect_trigger_tags(case_state):
    """
    Build the set of trigger tags implied by the case state.
    case_state is expected to look like:
        {
            "suspected_pathology": ["acl_tear"],   # confirmed/suspected only
            "ruled_out": ["meniscus"],             # explicitly excluded
            "phase": "post_surgical",              # acute | conservative |
                                                   # post_surgical |
                                                   # progress_tracking |
                                                   # return_to_sport
            "patient_type": "athlete",             # athlete | sedentary | elderly
            "primary_joint": "knee",
        }
    """

    tags = set()

    suspected = set(case_state.get("suspected_pathology", []))
    ruled_out = set(case_state.get("ruled_out", []))

    # Guard: never act on a pathology that has been ruled out, even if it
    # also appears in suspected (ruled_out wins).
    active = suspected - ruled_out

    phase = case_state.get("phase")
    patient_type = case_state.get("patient_type")

    # --- Rule: no suggestions at all during acute assessment ---
    # Outcome measures are for tracking, not triage. If the case is still
    # acute, return no tags so nothing fires.
    if phase == "acute" or phase is None:
        return set()

    # --- ACL-related tags ---
    if "acl_tear" in active or "acl" in active:
        tags.add("acl_injury_conservative")
        if phase == "post_surgical":
            tags.add("post_acl_reconstruction")
            tags.add("post_knee_ligament_surgery")

    # --- Instability / chondral tracking ---
    if "instability" in active:
        tags.add("clinical_instability_tracking")
    if "chondral_lesion" in active:
        tags.add("chondral_lesion")

    # --- Patellofemoral tags ---
    if "patellofemoral" in active or "pfps" in active:
        tags.add("patellofemoral_pain_syndrome")
    if "chondromalacia" in active:
        tags.add("chondromalacia_patellae")

    # --- Generic progress tracking ---
    if phase == "progress_tracking":
        tags.add("progress_tracking")

    # --- Return-to-sport tags: only for athletes, and only once the case
    # has reached a return-to-sport or progress-tracking phase. Deliberately
    # gated so sport measures don't fire in early post-surgical care. ---
    if patient_type == "athlete" and phase in ("progress_tracking", "return_to_sport"):
        tags.add("return_to_sport_decision")
        tags.add("activity_level_comparison")
        tags.add("rehabilitation_goal_setting")

    return tags


# -----------------------------------------------------------
# Main entry point
# -----------------------------------------------------------

def suggest_outcome_measures(case_state):
    """
    Return a list of outcome-measure suggestion dicts appropriate to the
    given case state. Empty list during acute assessment or when nothing
    matches.

    Each returned dict is safe to render directly in the UI's "suggested"
    (advisory) visual register -- NOT the deterministic "system check"
    register, because a recommended outcome measure is advice, not a hard
    rule like an Ottawa positive.
    """

    if not isinstance(case_state, dict):
        return []

    tags = _collect_trigger_tags(case_state)
    if not tags:
        return []

    phase = case_state.get("phase")

    suggestions = []

    for key, measure in OUTCOME_MEASURES.items():

        # Does this measure's when_to_use overlap the triggered tags?
        if not tags.intersection(set(measure["when_to_use"])):
            continue

        # Phase-appropriateness guard: skip measures that are not
        # appropriate for the current rehab phase (e.g. Tegner
        # return-to-sport grading in early post-surgical care).
        phase_ok = measure.get("phase_appropriate")
        if phase_ok is not None and phase is not None and phase not in phase_ok:
            continue

        suggestions.append({
            "key": key,
            "name": measure["name"],
            "scope": measure["scope"],
            "scoring": measure["scoring"],
            "source": measure["source"],
            # Marks this as advisory for the UI layer.
            "register": "suggested",
        })

    return suggestions


# -----------------------------------------------------------
# Quick manual test
# Run: python -m backend.safety.outcome_trigger
# -----------------------------------------------------------

if __name__ == "__main__":

    test_cases = [
        # Acute -> nothing should fire.
        {
            "label": "acute ACL, still triaging",
            "state": {
                "suspected_pathology": ["acl_tear"],
                "phase": "acute",
                "patient_type": "athlete",
            },
        },
        # Post-surgical ACL athlete -> IKDC, Lysholm (not Tegner, too early).
        {
            "label": "post-surgical ACL athlete",
            "state": {
                "suspected_pathology": ["acl_tear"],
                "phase": "post_surgical",
                "patient_type": "athlete",
            },
        },
        # Progress-tracking ACL athlete -> IKDC, Lysholm, Tegner.
        {
            "label": "progress-tracking ACL athlete",
            "state": {
                "suspected_pathology": ["acl_tear"],
                "phase": "progress_tracking",
                "patient_type": "athlete",
            },
        },
        # ACL ruled out -> nothing should fire on ACL.
        {
            "label": "ACL ruled out",
            "state": {
                "suspected_pathology": ["acl_tear"],
                "ruled_out": ["acl_tear"],
                "phase": "post_surgical",
                "patient_type": "athlete",
            },
        },
        # Patellofemoral case -> Kujala.
        {
            "label": "patellofemoral, conservative",
            "state": {
                "suspected_pathology": ["patellofemoral"],
                "phase": "conservative",
                "patient_type": "sedentary",
            },
        },
    ]

    for case in test_cases:
        print("=" * 60)
        print("CASE:", case["label"])
        results = suggest_outcome_measures(case["state"])
        if not results:
            print("  (no suggestions)")
        for r in results:
            print("  ->", r["name"])

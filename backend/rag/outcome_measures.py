# outcome_measures.py
#
# Reference data for knee outcome-measurement instruments.
#
# These are deliberately NOT embedded in ChromaDB. Outcome measures are
# rehabilitation / progress-tracking instruments, not acute-assessment
# tools. Embedding them in the vector store caused them to surface on
# acute queries (e.g. an Ottawa-rule query returned outcome-scale
# fragments at ~0.55 relevance), polluting retrieval. Instead, they are
# held here as structured data and surfaced only by a deterministic
# trigger (see backend/safety/outcome_trigger.py), which fires on the
# accumulated case state rather than on query similarity.
#
# Licensed instruments (IKDC, Tegner, etc.) are included under
# institutional educational-use terms confirmed by the supervisor: the
# system is not published, remains on college-controlled infrastructure,
# and is not distributed. The full instrument content is held in the
# college-internal copy; the "source" field below records where each
# instrument appears in that copy.

# Each entry is one instrument. The "when_to_use" list holds the trigger
# tags that the trigger function matches the case state against.

OUTCOME_MEASURES = {

    "ikdc_subjective": {
        "name": "2000 IKDC Subjective Knee Evaluation Form",
        "when_to_use": [
            "post_acl_reconstruction",
            "acl_injury_conservative",
            "post_knee_ligament_surgery",
            "progress_tracking",
        ],
        "scope": "Symptoms (pain, stiffness, swelling, locking, giving-way), "
                 "sports and daily activities, current knee function.",
        "scoring": "18 items transformed to a 0-100 scale; higher = better "
                   "function and fewer symptoms.",
        "phase_appropriate": ["conservative", "post_surgical", "progress_tracking"],
        "source": "College-internal copy (cf. Magee Ch. 12, Fig. 12-45).",
    },

    "lysholm": {
        "name": "Lysholm Scoring Scale",
        "when_to_use": [
            "post_acl_reconstruction",
            "clinical_instability_tracking",
            "chondral_lesion",
            "progress_tracking",
        ],
        "scope": "Limp, support, stair-climbing, squatting, instability, "
                 "pain, swelling, thigh atrophy.",
        "scoring": "8 items, 0-100 scale; higher = better function.",
        "phase_appropriate": ["conservative", "post_surgical", "progress_tracking"],
        "source": "College-internal copy (cf. Magee Ch. 12, Table 12-7).",
    },

    "tegner": {
        "name": "Tegner Activity Level Scale",
        "when_to_use": [
            "return_to_sport_decision",
            "activity_level_comparison",
            "rehabilitation_goal_setting",
        ],
        "scope": "Activity level 0-10, from sick leave/disability to elite "
                 "competitive sport. Often paired with Lysholm.",
        "scoring": "Single graded level 0-10 (pre-injury vs current).",
        # Deliberately NOT phase-appropriate in early post-surgical care:
        # return-to-sport grading is premature immediately after surgery.
        "phase_appropriate": ["conservative", "progress_tracking", "return_to_sport"],
        "source": "College-internal copy (cf. Magee Ch. 12, Fig. 12-48).",
    },

    "kujala": {
        "name": "Kujala Patellofemoral Score",
        "when_to_use": [
            "patellofemoral_pain_syndrome",
            "patellar_subluxation_post_op",
            "chondromalacia_patellae",
        ],
        "scope": "Patellofemoral-specific functional outcomes.",
        "scoring": "0-100 scale; higher = better function.",
        "phase_appropriate": ["conservative", "post_surgical", "progress_tracking"],
        "source": "College-internal copy (cf. Magee Ch. 12, Table 12-8).",
    },
}

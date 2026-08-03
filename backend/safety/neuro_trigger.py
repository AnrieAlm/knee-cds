# backend/safety/neuro_trigger.py
# deterministic neuro screen trigger — no LLM involved
# returns (triggered: 'yes'/'no', reason: str)

def check_neuro_trigger(physical: dict) -> tuple:

    # manual override — clinician explicitly requested it
    if physical.get('neuro_screen_requested') == 'yes':
        return 'yes', 'clinician requested neuro screen'

    # altered sensation checkbox in observation
    if physical.get('altered_sensation_reported') == 'yes':
        return 'yes', 'altered sensation reported in observation'

    # any MMT grade <= 2
    mmt_fields = [
        'mmt_quadriceps', 'mmt_hamstrings', 'mmt_hip_flexors',
        'mmt_hip_abductors', 'mmt_hip_external_rotators', 'mmt_gastroc_soleus'
    ]
    for field in mmt_fields:
        val = physical.get(field, '')
        if val and val.isdigit() and int(val) <= 2:
            return 'yes', f'muscle grade {val}/5 recorded for {field}'

    # positive posterior drawer implicates peroneal nerve
    if physical.get('test_posterior_drawer') == 'positive':
        return 'yes', 'positive posterior drawer — PCL involvement, check peroneal nerve'

    return 'no', ''
# backend/safety/neuro_trigger.py
# deterministic neuro screen trigger — no LLM involved
# returns (triggered: 'yes'/'no', reason: str)

# the muscles checked for weakness, in the order they are reported
mmt_fields = [
    'mmt_quadriceps', 'mmt_hamstrings', 'mmt_hip_flexors',
    'mmt_hip_abductors', 'mmt_hip_external_rotators', 'mmt_gastroc_soleus'
]

# a grade limited by pain or effusion is not evidence of neurological weakness
# a muscle marked not tested has no grade to interpret at all
# these are the limiter values that suppress the weakness trigger
suppressing_limiters = ['pain', 'effusion', 'not_tested']


def check_neuro_trigger(physical: dict) -> tuple:

    # manual override — clinician explicitly requested it
    if physical.get('neuro_screen_requested') == 'yes':
        return 'yes', 'clinician requested neuro screen'

    # altered sensation checkbox in observation
    if physical.get('altered_sensation_reported') == 'yes':
        return 'yes', 'altered sensation reported in observation'

    # any MMT grade <= 2, unless the grade was limited by pain or effusion
    # a quadriceps graded 2/5 because the patient is guarding is a different
    # finding from a quadriceps graded 2/5 because the femoral nerve is involved
    # firing the neuro screen on the first is a false positive inside the
    # deterministic layer, so the limiter has to be read before the grade
    for field in mmt_fields:
        val = physical.get(field, '')
        if val and str(val).isdigit() and int(val) <= 2:
            limiter = physical.get(field + '_limiter', '')
            if limiter in suppressing_limiters:
                continue
            return 'yes', f'muscle grade {val}/5 recorded for {field}'

    # positive posterior drawer implicates peroneal nerve
    if physical.get('test_posterior_drawer') == 'positive':
        return 'yes', 'positive posterior drawer — PCL involvement, check peroneal nerve'

    return 'no', ''


# a low grade that was suppressed above still matters clinically, it just is not
# a neurological finding yet — the agent should be told to re-test when the knee
# settles rather than told nothing at all
# this is read by physical_context.buildNeuroText via the inhibition_noted field
def check_inhibition(physical: dict) -> str:
    for field in mmt_fields:
        val = physical.get(field, '')
        if val and str(val).isdigit() and int(val) <= 2:
            limiter = physical.get(field + '_limiter', '')
            if limiter in ['pain', 'effusion']:
                return 'yes'
    return ''
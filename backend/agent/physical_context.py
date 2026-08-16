# backend/agent/physical_context.py
# turns the stored physical block into a short context string for the ReAct agent
# brevity used to be the whole point, because num_ctx was 2048 on the local model and the
# ReAct loop resends the accumulated context on every iteration
# the hosted model has a much larger window, so the budget below is raised and the trimming
# rarely fires — it is kept because a context that silently grows without limit is a
# governance problem, not because it is currently binding

from backend.validation.rom_plausibility import buildPlausibilityWarningText
from backend.constants import (
    DEFERRAL_REASON_LABELS,
    DEFERRAL_RETRY_CONDITION,
)
# the muscles in the order they should be reported
context_muscle_order = [
    ('quadriceps', 'quadriceps'),
    ('hamstrings', 'hamstrings'),
    ('hip_abductors', 'hip abductors'),
    ('hip_external_rotators', 'hip ext rotators'),
    ('hip_flexors', 'hip flexors'),
    ('gastroc_soleus', 'gastroc/soleus')
]

# short labels for the end feels, the stored values are too long to spend tokens on
end_feel_short_labels = {
    'tissue_approximation': 'tissue approximation',
    'tissue_stretch': 'tissue stretch',
    'capsular': 'capsular',
    'springy_block': 'springy block',
    'spasm': 'spasm',
    'bone_to_bone': 'bone to bone',
    'empty_feel': 'empty'
}

# the special tests the physio can record, in the order they should be reported
# grouped by structure so the agent reads related tests together
special_test_labels = [
    ('test_lachman', 'Lachman'),
    ('test_anterior_drawer', 'anterior drawer'),
    ('test_pivot_shift', 'pivot shift'),
    ('test_posterior_drawer', 'posterior drawer'),
    ('test_sag_sign', 'sag sign'),
    ('test_valgus_0', 'valgus stress 0'),
    ('test_valgus_30', 'valgus stress 30'),
    ('test_varus_0', 'varus stress 0'),
    ('test_varus_30', 'varus stress 30'),
    ('test_mcmurray', 'McMurray'),
    ('test_thessaly', 'Thessaly'),
    ('test_apley_compression', 'Apley compression'),
    ('test_apley_distraction', 'Apley distraction'),
    ('test_patellar_apprehension', 'patellar apprehension')
]

# bony points, only reported when actually tender
bony_point_labels = [
    ('patellar_tenderness', 'patella'),
    ('fibular_head_tenderness', 'fibular head'),
    ('tibial_tubercle_tenderness', 'tibial tubercle'),
    ('popliteal_tenderness', 'popliteal fossa')
]

# values that mean nothing was recorded, kept in one place because the empty check
# is repeated in every builder below
not_recorded_values = ['', None, 'not_assessed', 'not_done']


# a stored value is only a finding if it is not one of the not-recorded markers
# ROM fields are stored as int by the router, so this must not assume a string
def hasValue(physical_dict, field):
    return physical_dict.get(field) not in not_recorded_values


# rough token estimate, about four characters per token for english text
# this is only a guard rail, not an accurate count
def estimateTokens(text):
    return int(len(text) / 4) + 1


# build the range of motion sentence
# only findings that were actually recorded are mentioned, an unrecorded field is skipped
# entirely rather than reported as not recorded
#
# FIELD NAMES: these previously read rom_flexion_involved_active and
# rom_extension_involved_active, which no version of the form has ever posted. The whole
# flexion branch therefore always failed and the degrees never reached the agent. The
# failure was invisible because an absent field and an unrecorded finding take the same
# code path.
#
# TYPES: ROM is stored as int, so every numeric value is wrapped in str() before it is
# concatenated. Without this the first flexion reading raises TypeError inside the
# background agent thread, which surfaces only as an error status on the suggestion.
def buildRangeText(physical_dict):
    parts = []

    involved_flexion = physical_dict.get('rom_flexion_involved')
    uninvolved_flexion = physical_dict.get('rom_flexion_uninvolved')

    # flexion is reported as a pair because the uninvolved side is the reference
    if involved_flexion not in not_recorded_values and uninvolved_flexion not in not_recorded_values:
        flexion_text = 'flexion ' + str(involved_flexion) + '/' + str(uninvolved_flexion)

        # the deficit is computed here rather than stored, because it is derived from two
        # numbers that are already present and a stored copy could go stale
        try:
            deficit = int(uninvolved_flexion) - int(involved_flexion)
            if deficit > 0:
                flexion_text = flexion_text + ' (' + str(deficit) + ' degree deficit)'
        except (ValueError, TypeError):
            pass

        parts.append(flexion_text)
    elif involved_flexion not in not_recorded_values:
        parts.append('flexion ' + str(involved_flexion))

    involved_extension = physical_dict.get('rom_extension_involved')
    if involved_extension not in not_recorded_values:
        parts.append('extension ' + str(involved_extension))

    # rotation is where the meniscal tests live — McMurray, Apley and Thessaly are
    # all rotation tests, so a restricted or painful rotation directly bears on
    # which of them are worth suggesting
    # a full rotation is a pertinent negative but not worth the tokens at this stage
    rotation_labels = [
        ('rotation_medial', 'medial rotation'),
        ('rotation_lateral', 'lateral rotation')
    ]
    for field, label in rotation_labels:
        value = physical_dict.get(field, '')
        if value not in not_recorded_values and value != 'normal':
            parts.append(label + ' ' + value)

    # only the positive toggles earn a mention, a no is dropped to save tokens
    if physical_dict.get('extension_lag', '') == 'yes':
        parts.append('extension lag')

    # the form records pain as one field with four values rather than two booleans
    pain_on_movement = physical_dict.get('pain_on_movement', '')
    if pain_on_movement == 'flexion':
        parts.append('pain on flexion')
    elif pain_on_movement == 'extension':
        parts.append('pain on extension')
    elif pain_on_movement == 'both':
        parts.append('pain on flexion and extension')

    # the Ottawa criterion is recorded explicitly rather than inferred from the degrees,
    # because a blank reading is not the same as an inability to reach 90
    if physical_dict.get('able_to_flex_90', '') == 'no':
        parts.append('unable to flex to 90 degrees')

    # end feel is a passive finding, reported when recorded
    end_feel_flexion = physical_dict.get('end_feel_flexion', '')
    if end_feel_flexion not in not_recorded_values:
        parts.append('flexion end feel ' + end_feel_short_labels.get(end_feel_flexion, end_feel_flexion))

    end_feel_extension = physical_dict.get('end_feel_extension', '')
    if end_feel_extension not in not_recorded_values:
        parts.append('extension end feel ' + end_feel_short_labels.get(end_feel_extension, end_feel_extension))

    # nothing about range was recorded, so say nothing at all
    if len(parts) == 0:
        return ''

    return 'ROM: ' + ', '.join(parts) + '.'

# build the sentence for the deterministic flags
# these are computed by rule rather than observed, so they are stated as findings the agent
# may rely on rather than as opinions it should re-derive
#
# the pattern flags this used to read (capsular_pattern_flag, meniscus_rom_pattern,
# terminal_extension_achieved) are computed nowhere in the codebase, so this builder has
# always returned an empty string. They are removed rather than left reading fields that
# do not exist. Reinstate them alongside the module that computes them.
def buildFlagText(physical_dict):
    parts = []

    # the Cyriax sequence maps to an acuity stage deterministically
    sequence = physical_dict.get('pain_resistance_sequence', '')
    if sequence == 'pain_before_resistance':
        parts.append('acute presentation on the pain-resistance sequence')
    elif sequence == 'simultaneous':
        parts.append('subacute presentation on the pain-resistance sequence')
    elif sequence == 'resistance_before_pain':
        parts.append('chronic presentation on the pain-resistance sequence')

    if len(parts) == 0:
        return ''

    return 'Rule-based findings: ' + '; '.join(parts) + '.'


# build the observation and palpation sentence
# this section did not exist before, so gait, effusion and joint line tenderness have
# never reached the agent despite being recorded on the first page of the examination
def buildObservationText(physical_dict):
    parts = []

    gait = physical_dict.get('gait', '')
    if gait not in not_recorded_values and gait != 'normal':
        parts.append('gait ' + gait.replace('_', ' '))

    effusion = physical_dict.get('effusion', '')
    if effusion not in not_recorded_values and effusion != 'none':
        effusion_text = effusion + ' effusion'
        effusion_type = physical_dict.get('effusion_type', '')
        if effusion_type not in not_recorded_values and effusion_type != 'unknown':
            effusion_text = effusion_text + ' (' + effusion_type + ')'
        parts.append(effusion_text)

    joint_line = physical_dict.get('joint_line_tenderness', '')
    if joint_line not in not_recorded_values and joint_line != 'none':
        parts.append(joint_line + ' joint line tenderness')

    collateral = physical_dict.get('collateral_tenderness', '')
    if collateral not in not_recorded_values and collateral != 'none':
        parts.append(collateral.upper() + ' line tenderness')

    # bony tenderness feeds the fracture rules, which have already run, but it also
    # changes which special tests are reasonable to perform
    tender_points = []
    for field, label in bony_point_labels:
        if physical_dict.get(field, '') == 'yes':
            tender_points.append(label)
    if len(tender_points) > 0:
        parts.append('tenderness at ' + ', '.join(tender_points))

    temperature = physical_dict.get('joint_temperature', '')
    if temperature not in not_recorded_values and temperature != 'normal':
        parts.append('joint ' + temperature + ' to touch')

    if len(parts) == 0:
        return ''

    return 'Observation: ' + ', '.join(parts) + '.'


# build the strength sentence
# a muscle that was never graded is skipped, a limiter is only named when it changes the meaning
def buildStrengthText(physical_dict):
    parts = []

    for field, label in context_muscle_order:
        grade = physical_dict.get('mmt_' + field, '')

        # nothing graded for this muscle
        if grade in not_recorded_values:
            continue

        limiter = physical_dict.get('mmt_' + field + '_limiter', '')

        # a muscle marked not tested is reported as such rather than as a grade
        if limiter == 'not_tested':
            parts.append(label + ' not tested')
            continue

        grade_text = label + ' ' + str(grade) + '/5'

        # pain and effusion limited grades must not be read as neurological weakness
        if limiter in ['pain', 'effusion']:
            grade_text = grade_text + ' (' + limiter + ' limited)'

        parts.append(grade_text)

    if len(parts) == 0:
        return ''

    return 'Strength: ' + ', '.join(parts) + '.'


# build the special tests sentence
# this is the main reason physical findings are sent to the agent at all — a test that has
# already been performed should not be suggested again, and its result changes what is
# worth doing next
# build the special tests sentences
# this is the main reason physical findings are sent to the agent at all — a test that has
# already been performed should not be suggested again, and its result changes what is
# worth doing next
#
# THREE states are now distinguished, because they mean different things to the agent:
#
#   recorded (positive/negative)  performed. Excluded from suggestion, given as evidence.
#   deferred                      indicated but not performed. The agent should know the
#                                 test is outstanding AND why, because the reason often
#                                 constrains what else is worth suggesting — a knee too
#                                 swollen for McMurray is also too swollen for Thessaly.
#   not_indicated                 a closed clinical judgement. Deliberately NOT sent.
#                                 Re-raising a decision the clinician already made is how
#                                 a decision aid trains people to ignore it.
#   unset                         never addressed. Silent here; the deterministic coverage
#                                 layer surfaces it, not the model.
#
# Previously all four collapsed into one line reading "Tests already performed: Lachman
# deferred", which told the model the opposite of the truth under a heading that
# contradicted its own value, and dropped the reason entirely.
def buildSpecialTestText(physical_dict, deferrals=None):
    deferrals = deferrals or {}
    performed = []
    deferred = []

    for field, label in special_test_labels:
        result = physical_dict.get(field, '')

        if result == 'deferred':
            entry = deferrals.get(field) or {}
            reason_key = entry.get('reason', '')
            reason = DEFERRAL_REASON_LABELS.get(reason_key, '')
            retry = DEFERRAL_RETRY_CONDITION.get(reason_key, '')

            detail = ''
            if reason and retry:
                detail = ' (' + reason.lower() + ' — ' + retry.rstrip('.').lower() + ')'
            elif reason:
                detail = ' (' + reason.lower() + ')'

            deferred.append(label + detail)
            continue

        # not_indicated is a closed decision and is withheld from the agent
        # entirely. It is not evidence and it is not an outstanding task.
        if result == 'not_indicated':
            continue

        # unset, legacy not_done, and the other not-recorded markers
        if result in not_recorded_values:
            continue

        performed.append(label + ' ' + result)

    sentences = []
    if performed:
        sentences.append('Tests already performed: ' + ', '.join(performed) + '.')
    if deferred:
        sentences.append(
            'Tests indicated but not performed: ' + ', '.join(deferred) + '.'
        )

    return ' '.join(sentences)


# build the neuro sentence
# inside a triggered screen the normal findings are kept, because an intact dermatome is a
# pertinent negative that changes the interpretation rather than noise
def buildNeuroText(physical_dict):
    if physical_dict.get('neuro_triggered', '') != 'yes':
        # a suppressed trigger still matters, the agent should suggest re-testing
        if physical_dict.get('inhibition_noted', '') == 'yes':
            return 'Neuro screen not triggered; weakness attributed to pain or effusion, re-test when settled.'
        return ''

    parts = ['triggered because ' + physical_dict.get('neuro_trigger_reason', 'unspecified')]

    dermatomes = [('sensation_l3', 'L3'), ('sensation_l4', 'L4'),
                  ('sensation_l5', 'L5'), ('sensation_s1', 'S1')]

    sensation_parts = []
    for field, label in dermatomes:
        value = physical_dict.get(field, '')
        if value not in not_recorded_values:
            sensation_parts.append(label + ' ' + value)

    if len(sensation_parts) > 0:
        parts.append('sensation ' + ', '.join(sensation_parts))

    reflex_patella = physical_dict.get('reflex_patella', '')
    if reflex_patella not in not_recorded_values:
        parts.append('patellar reflex ' + reflex_patella)

    reflex_achilles = physical_dict.get('reflex_achilles', '')
    if reflex_achilles not in not_recorded_values:
        parts.append('achilles reflex ' + reflex_achilles)

    peroneal = physical_dict.get('peroneal_dorsiflexion', '')
    if peroneal not in not_recorded_values:
        parts.append('dorsiflexion ' + peroneal)

    return 'Neuro screen ' + '; '.join(parts) + '.'


# deferrals is passed separately because it lives on the case document rather
# than inside physical. A deferral is a statement ABOUT a finding, not a
# finding, so it is stored alongside rather than within - see constants.py.
# Defaulted to None so any caller not yet updated keeps working, just without
# the deferral line.
def formatPhysicalForAgent(physical_dict, token_budget=650, deferrals=None):
    # an empty or missing block tells the agent to suggest an examination rather than interpret one
    if not physical_dict:
        return 'Physical examination not yet recorded.'

    neuro_text = buildNeuroText(physical_dict)
    neuro_is_triggered = (physical_dict.get('neuro_triggered', '') == 'yes')

    # each section carries the order it should be read in and how hard it is to give up
    # a lower keep_rank is dropped later, so a triggered neuro screen is the last thing to go
    # completed tests rank second because re-suggesting a test the physio has already done
    # is the most visible failure this context is meant to prevent
    # read_order 0 puts the warning before any finding it applies to; keep_rank 0
    # makes it the one section that is never traded away for tokens. A context
    # trimmed down to findings with the warning about those findings removed would
    # be worse than no context at all.
    sections = [
        {'read_order': 0, 'keep_rank': 0,
         'text': buildPlausibilityWarningText(physical_dict)},
        {'read_order': 1, 'keep_rank': 5, 'text': buildObservationText(physical_dict)},
        {'read_order': 2, 'keep_rank': 4, 'text': buildRangeText(physical_dict)},
        {'read_order': 3, 'keep_rank': 3, 'text': buildFlagText(physical_dict)},
        {'read_order': 4, 'keep_rank': 6, 'text': buildStrengthText(physical_dict)},
        {'read_order': 5, 'keep_rank': 2, 'text': buildSpecialTestText(physical_dict, deferrals)},
        {'read_order': 6, 'keep_rank': 1 if neuro_is_triggered else 7, 'text': neuro_text}
    ]

    # drop the sections that produced nothing
    kept_sections = []
    for section in sections:
        if section['text'] != '':
            kept_sections.append(section)

    if len(kept_sections) == 0:
        return 'Physical examination not yet recorded.'

    # assemble in reading order, which is not the same as the order things get dropped in
    def assemble(section_list):
        in_order = sorted(section_list, key=lambda item: item['read_order'])
        pieces = []
        for item in in_order:
            pieces.append(item['text'])
        return ' '.join(pieces)

    text = assemble(kept_sections)

    # over budget, so give up the highest keep_rank first
    while estimateTokens(text) > token_budget and len(kept_sections) > 1:
        droppable = [s for s in kept_sections if s['keep_rank'] > 0]
        if len(droppable) == 0:
            break
        worst = max(droppable, key=lambda item: item['keep_rank'])
        kept_sections.remove(worst)
        text = assemble(kept_sections)
        print('physical context trimmed, dropped a section to fit the token budget')

    print('physical context (' + str(estimateTokens(text)) + ' est tokens):', text)
    return text
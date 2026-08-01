# backend/agent/physical_context.py
# turns the stored physical block into a short context string for the ReAct agent
# the whole point of this module is brevity, because num_ctx is 2048 and the ReAct loop resends
# the accumulated context on every iteration


# the muscles in the order they should be reported, matching muscle_list in main.py
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


# rough token estimate, about four characters per token for english text
# this is only a guard rail, not an accurate count
def estimateTokens(text):
    return int(len(text) / 4) + 1


# build the range of motion sentence
# only findings that were actually recorded are mentioned, an unrecorded field is skipped
# entirely rather than reported as not recorded
def buildRangeText(physical_dict):
    parts = []

    involved_flexion = physical_dict.get('rom_flexion_involved_active', '')
    uninvolved_flexion = physical_dict.get('rom_flexion_uninvolved', '')
    deficit = physical_dict.get('rom_flexion_deficit_percent', '')

    # flexion is reported as a pair because the uninvolved side is the reference
    if involved_flexion != '' and uninvolved_flexion != '':
        flexion_text = 'flexion ' + involved_flexion + '/' + uninvolved_flexion
        if deficit != '':
            flexion_text = flexion_text + ' (' + deficit + '% deficit)'
        parts.append(flexion_text)
    elif involved_flexion != '':
        parts.append('flexion ' + involved_flexion)

    involved_extension = physical_dict.get('rom_extension_involved_active', '')
    if involved_extension != '':
        parts.append('extension ' + involved_extension)

    # a movement that could not be tested is a different signal from a low reading, so say so
    flexion_unable = physical_dict.get('rom_flexion_unable', '')
    if flexion_unable not in ['', 'no']:
        parts.append('flexion not testable due to ' + flexion_unable)

    extension_unable = physical_dict.get('rom_extension_unable', '')
    if extension_unable not in ['', 'no']:
        parts.append('extension not testable due to ' + extension_unable)

    # only the positive toggles earn a mention, a no is dropped to save tokens
    if physical_dict.get('extension_lag', '') == 'yes':
        parts.append('extension lag')

    if physical_dict.get('painful_arc', '') == 'yes':
        parts.append('painful arc')

    if physical_dict.get('pain_on_flexion', '') == 'yes':
        parts.append('pain on flexion')

    if physical_dict.get('pain_on_extension', '') == 'yes':
        parts.append('pain on extension')

    # nothing about range was recorded, so say nothing at all
    # without this guard the passive note below would invent a range section out of an empty exam
    if len(parts) == 0:
        return ''

    # passive findings only exist when passive range was assessed
    if physical_dict.get('rom_passive_recorded', '') == 'yes':
        end_feel_flexion = physical_dict.get('end_feel_flexion', '')
        if end_feel_flexion != '':
            parts.append('flexion end feel ' + end_feel_short_labels.get(end_feel_flexion, end_feel_flexion))

        end_feel_extension = physical_dict.get('end_feel_extension', '')
        if end_feel_extension != '':
            parts.append('extension end feel ' + end_feel_short_labels.get(end_feel_extension, end_feel_extension))
    else:
        # the agent needs to know this is an active only examination before it interprets anything
        # this only makes sense because the guard above proved some active range was recorded
        parts.append('passive range not assessed')

    return 'ROM: ' + ', '.join(parts) + '.'


# build the sentence for the deterministic flags
# these are computed by rule rather than observed, so they are stated as findings the agent
# may rely on rather than as opinions it should re-derive
def buildFlagText(physical_dict):
    parts = []

    if physical_dict.get('terminal_extension_achieved', '') == 'no':
        parts.append('terminal extension not achieved')

    if physical_dict.get('capsular_pattern_flag', '') == 'yes':
        parts.append('range pattern consistent with a capsular pattern')

    if physical_dict.get('meniscus_rom_pattern', '') == 'yes':
        parts.append('range loss matches the pattern listed for meniscus injury')

    irritability = physical_dict.get('irritability_stage', '')
    if irritability != '':
        parts.append(irritability + ' presentation on the pain-resistance sequence')

    if len(parts) == 0:
        return ''

    return 'Rule-based findings: ' + '; '.join(parts) + '.'


# build the strength sentence
# a muscle that was never graded is skipped, a limiter is only named when it changes the meaning
def buildStrengthText(physical_dict):
    parts = []

    for field, label in context_muscle_order:
        grade = physical_dict.get('mmt_' + field, '')

        # nothing graded for this muscle
        if grade == '':
            continue

        limiter = physical_dict.get('mmt_' + field + '_limiter', '')

        # a muscle marked not tested is reported as such rather than as a grade
        if limiter == 'not_tested':
            parts.append(label + ' not tested')
            continue

        grade_text = label + ' ' + grade + '/5'

        # pain and effusion limited grades must not be read as neurological weakness
        if limiter in ['pain', 'effusion']:
            grade_text = grade_text + ' (' + limiter + ' limited)'

        parts.append(grade_text)

    if len(parts) == 0:
        return ''

    return 'Strength: ' + ', '.join(parts) + '.'


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
        if value != '':
            sensation_parts.append(label + ' ' + value)

    if len(sensation_parts) > 0:
        parts.append('sensation ' + ', '.join(sensation_parts))

    reflex_patella = physical_dict.get('reflex_patella', '')
    if reflex_patella != '':
        parts.append('patellar reflex ' + reflex_patella)

    peroneal = physical_dict.get('peroneal_dorsiflexion', '')
    if peroneal != '':
        parts.append('dorsiflexion ' + peroneal)

    return 'Neuro screen ' + '; '.join(parts) + '.'


# the function the orchestrator calls
# returns a short block, or a single short line when the examination has not been done yet
def formatPhysicalForAgent(physical_dict, token_budget=250):
    # an empty or missing block tells the agent to suggest an examination rather than interpret one
    if not physical_dict:
        return 'Physical examination not yet recorded.'

    neuro_text = buildNeuroText(physical_dict)
    neuro_is_triggered = (physical_dict.get('neuro_triggered', '') == 'yes')

    # each section carries the order it should be read in and how hard it is to give up
    # a lower keep_rank is dropped later, so a triggered neuro screen is the last thing to go
    sections = [
        {'read_order': 1, 'keep_rank': 3, 'text': buildRangeText(physical_dict)},
        {'read_order': 2, 'keep_rank': 2, 'text': buildFlagText(physical_dict)},
        {'read_order': 3, 'keep_rank': 4, 'text': buildStrengthText(physical_dict)},
        {'read_order': 4, 'keep_rank': 1 if neuro_is_triggered else 5, 'text': neuro_text}
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
        worst = max(kept_sections, key=lambda item: item['keep_rank'])
        kept_sections.remove(worst)
        text = assemble(kept_sections)
        print('physical context trimmed, dropped a section to fit the token budget')

    print('physical context (' + str(estimateTokens(text)) + ' est tokens):', text)
    return text
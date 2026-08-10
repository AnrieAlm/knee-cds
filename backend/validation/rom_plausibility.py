# backend/validation/rom_plausibility.py
#
# Cross-field physiological plausibility checks on recorded ROM.
#
# This module is ADVISORY ONLY. It never gates, never suppresses, never blocks a
# save, and never alters a stored value. It exists because the deterministic
# safety layer validates criteria satisfaction (does this record meet Ottawa's
# thresholds) and the agentic layer validates nothing at all — a language model
# conditioned on a structured context has no prior that the context should be
# internally consistent. A record stating 150 degrees of flexion alongside a
# 30 degree extension deficit and an extension lag describes a knee that cannot
# exist, and both layers process it without complaint.
#
# The checks are cross-field by design. Per-field range bounds cannot catch this
# class of error: every individual value in that record is inside its own legal
# range. The contradiction lives in the combination.
#
# Deliberately NOT placed in backend/safety/ — everything in that package
# produces a binding decision. This produces advice.


# Magee gives full knee flexion as 135 degrees. Values above that are recorded
# rather than rejected, because hypermobility is real, but they are flagged.
NORMAL_FULL_FLEXION = 135

# Gauge bounds from the ROM form, kept in sync deliberately.
FLEXION_MIN, FLEXION_MAX = 0, 150
EXTENSION_MIN, EXTENSION_MAX = -10, 30


def _coerce_int(value):
    """Return an int, or None if the value is absent or not numeric.

    ROM has been stored as both int and str across the life of this codebase,
    so this must not assume either.
    """
    if value is None or value == '':
        return None
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def checkRomPlausibility(physical_dict):
    """Return a list of human-readable warning strings. Empty list means no
    detected contradiction — which is not the same as 'the data is correct'."""
    if not physical_dict:
        return []

    warnings = []

    flex_i = _coerce_int(physical_dict.get('rom_flexion_involved'))
    flex_u = _coerce_int(physical_dict.get('rom_flexion_uninvolved'))
    ext_i = _coerce_int(physical_dict.get('rom_extension_involved'))
    ext_u = _coerce_int(physical_dict.get('rom_extension_uninvolved'))
    lag = physical_dict.get('extension_lag', '') == 'yes'
    flex_90 = physical_dict.get('able_to_flex_90', '')

    # --- single-field sanity, cheap and catches gauge pinning -------------
    if flex_i is not None and flex_i > NORMAL_FULL_FLEXION:
        warnings.append(
            'recorded involved-side flexion of ' + str(flex_i) +
            ' degrees exceeds normal full range (' + str(NORMAL_FULL_FLEXION) +
            ' degrees); verify the measurement or document hypermobility')

    if flex_u is not None and flex_u > NORMAL_FULL_FLEXION:
        warnings.append(
            'recorded uninvolved-side flexion of ' + str(flex_u) +
            ' degrees exceeds normal full range (' + str(NORMAL_FULL_FLEXION) +
            ' degrees)')

    # --- cross-field contradictions ---------------------------------------
    # the involved side exceeding the uninvolved side by a wide margin usually
    # means the two gauges were filled the wrong way round
    if flex_i is not None and flex_u is not None and flex_i > flex_u + 10:
        warnings.append(
            'involved-side flexion (' + str(flex_i) + ') exceeds uninvolved-side '
            'flexion (' + str(flex_u) + ') by more than 10 degrees; the sides may '
            'have been recorded the wrong way round')

    # a fixed flexion deformity restricts the total arc — a knee that will not
    # straighten by 10 degrees or more does not also reach near-full flexion
    if flex_i is not None and ext_i is not None and ext_i >= 10 and flex_i >= 130:
        warnings.append(
            'an extension deficit of ' + str(ext_i) + ' degrees is inconsistent '
            'with ' + str(flex_i) + ' degrees of flexion; a fixed flexion '
            'deformity restricts the total arc')

    # extension lag means the active range falls short of the passive range,
    # which cannot be true when the recorded range is essentially full
    if lag and (ext_i is None or ext_i <= 0):
        if flex_i is not None and flex_u is not None and (flex_u - flex_i) <= 5:
            warnings.append(
                'extension lag is recorded alongside a full or near-full range '
                'with no extension deficit; one of the two findings is likely '
                'to be in error')

    # the Ottawa criterion is recorded separately from the degrees, so the two
    # can disagree
    if flex_90 == 'no' and flex_i is not None and flex_i >= 90:
        warnings.append(
            'the Ottawa criterion records an inability to flex to 90 degrees, '
            'but ' + str(flex_i) + ' degrees of flexion is recorded')

    if flex_90 == 'yes' and flex_i is not None and flex_i < 90:
        warnings.append(
            'the Ottawa criterion records an ability to flex to 90 degrees, '
            'but only ' + str(flex_i) + ' degrees of flexion is recorded')

    # identical non-zero extension on both sides is more often a mis-tap on the
    # wrong gauge than a genuine bilateral deficit
    if ext_i is not None and ext_u is not None and ext_i == ext_u and ext_i > 0:
        warnings.append(
            'an identical extension deficit of ' + str(ext_i) + ' degrees is '
            'recorded on both sides; confirm this is a genuine bilateral finding')

    return warnings


# the string the agent sees, or '' when there is nothing to warn about
def buildPlausibilityWarningText(physical_dict):
    warnings = checkRomPlausibility(physical_dict)
    if len(warnings) == 0:
        return ''

    header = ('DATA QUALITY WARNING - the recorded findings below are '
              'physiologically inconsistent and cannot all be true. Do not '
              'reason as though they are. Recommend re-measurement rather than '
              'offering test suggestions that depend on these values: ')

    return header + '; '.join(warnings) + '.'
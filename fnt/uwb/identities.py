"""Animal identity labels shared by the UWB preview, exports and animation.

A tag's entry in ``tag_identities`` carries its sex and ID (the "SexID", e.g.
``F9905``) plus two optional, human-friendly fields:

    name  e.g. ``Hera``
    code  e.g. ``HER``

Everything that turns a tag into on-screen text goes through ``tag_label`` so
the preview, the rendered video and the tag list can never disagree about what
an animal is called. GUI-free on purpose: the completeness check below is what
the Configure Identities dialog runs before saving, and it is worth being able
to test without Qt.
"""

# Options for the "Show Tag ID" selector, in menu order.
SEX_ID = "SexID"
NAME = "Name"
CODE = "Code"
HEX_ID = "HexID"
SHORT_ID = "ShortID"
ID_DISPLAY_TYPES = (SEX_ID, NAME, CODE, HEX_ID, SHORT_ID)

# Earlier configs stored the SexID option under this name.
_LEGACY_TYPES = {"Display ID": SEX_ID}

# The optional free-text fields, as (dict key, human label).
OPTIONAL_FIELDS = (("name", NAME), ("code", CODE))


def normalize_id_type(id_type):
    """Map a stored/legacy selector value onto a current option."""
    id_type = _LEGACY_TYPES.get(id_type, id_type)
    return id_type if id_type in ID_DISPLAY_TYPES else SEX_ID


def hex_label(tag):
    """The tag's short address in hex, e.g. ``2A``."""
    return hex(int(tag)).upper().replace('0X', '')


def sex_id_label(info):
    """``F9905`` when both sex and ID are configured, else None."""
    info = info or {}
    sex = str(info.get('sex', '') or '').strip()
    ident = str(info.get('identity', '') or '').strip()
    return f"{sex}{ident}" if (sex and ident) else None


def field_value(info, key):
    """A stripped optional field (``name``/``code``), or '' when unset."""
    return str((info or {}).get(key, '') or '').strip()


def tag_label(tag, info, id_type=SEX_ID):
    """On-screen text for one tag.

    Name and Code are optional, so an animal without one falls back to its
    SexID rather than going unlabelled — a partially-filled roster still reads
    sensibly. SexID itself falls back to the hex address when the tag has no
    identity configured.
    """
    id_type = normalize_id_type(id_type)
    if id_type == SHORT_ID:
        return str(tag)
    if id_type == HEX_ID:
        return hex_label(tag)
    if id_type in (NAME, CODE):
        value = field_value(info, 'name' if id_type == NAME else 'code')
        if value:
            return value
    return sex_id_label(info) or hex_label(tag)


def animal_key(tag, info):
    """The animal a tag belongs to: tags sharing sex + ID are one animal.

    Matches how the dialog and the analyses merge a replaced tag into the
    animal it replaced.
    """
    info = info or {}
    sex = str(info.get('sex', '') or '').strip()
    ident = str(info.get('identity', '') or '').strip() or str(tag)
    return sex, ident


def identity_field_problems(identities):
    """Where the optional Name/Code fields are inconsistent across animals.

    Checked per ANIMAL, not per tag, so a replacement tag that shares its
    animal's ID does not need the name typed twice. For each field returns:

      missing     animals left blank while at least one other has a value
                  (a partly-filled roster, so a Name/Code view mixes names
                  with SexIDs)
      conflicts   one animal whose tags carry different values
      duplicates  one value used by more than one animal (the label would no
                  longer tell them apart)

    Only fields someone actually started filling in are reported; a roster
    with no names at all is complete, not missing them.
    """
    animals = {}                                   # key -> list of infos
    for tag, info in (identities or {}).items():
        animals.setdefault(animal_key(tag, info), []).append(info or {})

    def _name(key):
        sex, ident = key
        return f"{sex}{ident}" if sex else ident

    report = {}
    for field, label in OPTIONAL_FIELDS:
        values = {}                                # key -> set of values
        for key, infos in animals.items():
            values[key] = {v for v in (field_value(i, field) for i in infos) if v}
        if not any(values.values()):
            continue
        missing = sorted(_name(k) for k, v in values.items() if not v)
        conflicts = sorted((_name(k), sorted(v)) for k, v in values.items()
                           if len(v) > 1)
        # Case-insensitive: "Hera" and "hera" would still read as one label.
        owners = {}                                # casefold -> (shown, keys)
        for key, vals in values.items():
            for v in vals:
                owners.setdefault(v.casefold(), (v, set()))[1].add(key)
        duplicates = sorted((shown, sorted(_name(k) for k in keys))
                            for shown, keys in owners.values() if len(keys) > 1)
        if missing or conflicts or duplicates:
            report[field] = {'label': label, 'missing': missing,
                             'conflicts': conflicts, 'duplicates': duplicates}
    return report


def describe_problems(report, limit=12):
    """Human-readable lines for ``identity_field_problems`` output."""
    def _clip(items):
        items = list(items)
        more = len(items) - limit
        text = ", ".join(items[:limit])
        return text + (f", … (+{more} more)" if more > 0 else "")

    lines = []
    for field, _label in OPTIONAL_FIELDS:
        r = report.get(field)
        if not r:
            continue
        lbl = r['label']
        if r['missing']:
            lines.append(f"• No {lbl} for: {_clip(r['missing'])}")
        for animal, vals in r['conflicts']:
            lines.append(f"• {animal} has more than one {lbl}: "
                         f"{', '.join(vals)}")
        for val, owners in r['duplicates']:
            lines.append(f"• {lbl} “{val}” is used by "
                         f"{', '.join(owners)}")
    return lines

INPUT_SCHEMA = {
    'word_level_transcript': {
        'type': list,
        'required': True,
    },
    'image_count_hint': {
        'type': int,
        'required': True,
        "min": 1,
        "max": 10
    },
    'generation_steps': {
        'type': int,
        'required': False,
        "min": 10,
        "max": 50
    },
}

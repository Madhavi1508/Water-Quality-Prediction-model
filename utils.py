def validate_inputs(inputs, features):
    for value, feature in zip(inputs, features):
        if not (feature['min'] <= value <= feature['max']):
            return False
    return True
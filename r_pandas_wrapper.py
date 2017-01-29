def calculate_va_wrapper(json, *args):
    import pandas as pd
    from basic_va_alg import calculate_va
    return calculate_va(pd.read_json(json), *args)[0].to_json()

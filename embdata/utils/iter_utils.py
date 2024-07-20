
exists_iter = lambda key, c: c is not None and len(c) > 0 and (hasattr(c[0], key) or key in c[0])
get_iter = lambda key, c: None if not exists_iter(key, c) else c[0][key] if key in c[0] else getattr(c[0], key)
get_iter_class = lambda key, c: get_iter(key, c).__class__ if get_iter(key, c) is not None else None
get_iter_length = lambda key, c: len(get_iter(key, c)) if get_iter(key, c) is not None else None
get_iter_item = lambda key, c, i: get_iter(key, c)[i] if get_iter(key, c) is not None else None
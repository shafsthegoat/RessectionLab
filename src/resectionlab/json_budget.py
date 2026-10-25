"""Conservative acceptance proof for the stock compact ASCII JSON encoder.

False means unknown: callers must retain their ordinary streaming encoder and
its original exceptions. This is neither an exact size nor a rejection test.
"""
MAX_DEPTH = 64
MAX_NODES = 1_000_000
MAX_INT_BITS = 4096


def _scalar_bound(value, key=False):
    kind = type(value)
    if kind is str:
        return 12 * len(value) + 2  # Includes escaped surrogate pairs/quotes.
    if value is None:
        size = 4
    elif kind is bool:
        size = 4 if value else 5
    elif kind is int:
        if value.bit_length() > MAX_INT_BITS:
            return None  # Never allocate an unbounded decimal string.
        size = len(str(value))  # Digit-limit errors deliberately decline.
    elif kind is float:
        if not -1e309 < value < 1e309:
            return None
        size = 32  # Finite IEEE double repr, including sign/exponent/-0.
    else:
        return None
    return size + (2 if key else 0)  # Numeric/null/bool dict keys are quoted.


def builtin_json_fits(value, limit):
    """True proves stock dumps(ensure_ascii=True, allow_nan=False) fits limit.

    Only exact built-ins are inspected. Iterators retain at most one active
    path; aliases are charged on every appearance, cycles on that path decline.
    No object is changed and no custom conversion/iteration hook is called.
    """
    if type(limit) is not int or limit < 0:
        return False
    total = nodes = 0
    active = set()
    stack = [(None, iter((value,)), False)]
    try:
        while stack:
            marker, iterator, mapping = stack[-1]
            try:
                item = next(iterator)
            except StopIteration:
                stack.pop()
                if marker is not None:
                    active.remove(marker)
                continue
            nodes += 2 if mapping else 1
            if nodes > MAX_NODES:
                return False
            if mapping:
                key, item = item
                bound = _scalar_bound(key, True)
                if bound is None:
                    return False
                total += bound + 1  # Colon; comma already charged below.
            kind = type(item)
            if kind is list or kind is tuple or kind is dict:
                marker = id(item)
                if marker in active or len(stack) > MAX_DEPTH:
                    return False
                total += 2 + max(0, len(item) - 1)  # Brackets/braces/commas.
                active.add(marker)
                stack.append((marker, iter(item.items()) if kind is dict else iter(item), kind is dict))
            else:
                bound = _scalar_bound(item)
                if bound is None:
                    return False
                total += bound
            if total > limit:
                return False
        return True
    except (ValueError, OverflowError, RuntimeError):
        # Integer digit limits or concurrent builtin-container mutation. Let
        # the unchanged stock encoder handle the uncertain/error case.
        return False

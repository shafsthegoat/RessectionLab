"""The native bridge transports only explicitly supported outer checks."""
import pytest

from resectionlab import surgical_native_control as control


class Token:
    def __init__(self):
        self.timeouts = []

    def set_timeout(self, seconds):
        self.timeouts.append(seconds)


def test_deadline_is_absolute_shared_and_can_be_cleared(monkeypatch):
    monkeypatch.setattr(control, 'monotonic', lambda: 10.)
    token = Token()
    calls = []
    root = control.NativeCheck(lambda: calls.append('root'), token)
    parent = control.derive_native_check(root, lambda: calls.append('budget'), deadline=15.)
    child = control.derive_native_check(parent, lambda: calls.append('local'), deadline=18.)
    child()
    assert calls == ['local']
    assert control.native_stop_token(child) is token
    assert token.timeouts == [5.]
    assert control.native_stop_token(root) is token
    assert token.timeouts == [5., None]
    expired = control.derive_native_check(root, lambda: None, deadline=9.)
    control.native_stop_token(expired)
    assert token.timeouts[-1] == 0.


def test_combination_preserves_order_and_only_combines_one_token():
    calls, token = [], Token()
    first = control.NativeCheck(lambda: calls.append('meter'), token, deadline=20.)
    second = control.NativeCheck(lambda: calls.append('caller'), token, deadline=15.)

    def original():
        first()
        second()

    combined = control.combine_native_checks(first, second, original)
    assert type(combined) is control.NativeCheck
    assert combined._deadline == 15.
    combined()
    assert calls == ['meter', 'caller']
    other = control.NativeCheck(lambda: None, Token())
    assert control.combine_native_checks(first, other, original) is original


def test_arbitrary_checks_and_subclasses_keep_reference_behavior():
    calls = []
    callback = lambda: calls.append('custom')
    assert control.derive_native_check(callback, callback, deadline=10.) is callback
    assert control.native_stop_token(callback) is None
    known = control.NativeCheck(lambda: calls.append('known'), Token())

    def original():
        known()
        callback()

    combined = control.combine_native_checks(known, callback, original)
    assert combined is original
    assert control.native_stop_token(combined) is None
    combined()
    assert calls == ['known', 'custom']

    class CustomCheck(control.NativeCheck):
        pass

    assert control.native_stop_token(CustomCheck(callback, Token())) is None


@pytest.mark.parametrize('message', ['SIMULATION_CANCELLED', 'SIMULATION_DEADLINE_EXCEEDED'])
def test_stop_recovers_original_exception(message):
    expected = TimeoutError('original budget reason')

    def original():
        raise expected

    with pytest.raises(TimeoutError) as caught:
        control.translate_native_stop(control.NativeCheck(original, Token()), RuntimeError(message))
    assert caught.value is expected


@pytest.mark.parametrize('message', ['physics failure', 'SIMULATION_CANCELLED extra'])
def test_other_native_errors_never_run_the_outer_check(message):
    calls = []
    error = RuntimeError(message)
    with pytest.raises(RuntimeError) as caught:
        control.translate_native_stop(control.NativeCheck(lambda: calls.append(True), Token()), error)
    assert caught.value is error
    assert calls == []


def test_stop_without_an_outer_exception_remains_an_error():
    error = RuntimeError('SIMULATION_CANCELLED')
    with pytest.raises(RuntimeError) as caught:
        control.translate_native_stop(control.NativeCheck(lambda: None, Token()), error)
    assert caught.value is error


@pytest.mark.parametrize('deadline', [True, float('inf'), float('nan'), 'later'])
def test_invalid_deadlines_are_rejected(deadline):
    with pytest.raises(ValueError):
        control.NativeCheck(lambda: None, Token(), deadline=deadline)

from app.utils.metrics import calculate_accuracy, calculate_wpm


def test_calculate_wpm_standard_minute():
    # 250 characters = 50 words over one minute => 50 WPM
    assert calculate_wpm(chars_typed=250, duration_seconds=60) == 50.0


def test_calculate_wpm_zero_duration():
    assert calculate_wpm(chars_typed=120, duration_seconds=0) == 0.0


def test_calculate_accuracy_perfect():
    assert calculate_accuracy(correct_chars=10, total_chars=10) == 100.0


def test_calculate_accuracy_partial():
    assert calculate_accuracy(correct_chars=75, total_chars=100) == 75.0


def test_calculate_accuracy_no_chars_defaults_to_hundred():
    assert calculate_accuracy(correct_chars=0, total_chars=0) == 100.0

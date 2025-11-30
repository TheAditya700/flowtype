def calculate_wpm(chars_typed: int, duration_seconds: float) -> float:
    """
    Calculates Words Per Minute (WPM).
    Standard definition is (number of characters / 5) / 1 minute.
    """
    if duration_seconds == 0:
        return 0.0
    
    words_typed = chars_typed / 5.0
    minutes = duration_seconds / 60.0
    wpm = words_typed / minutes
    return wpm

def calculate_accuracy(correct_chars: int, total_chars: int) -> float:
    """
    Calculates accuracy as a percentage.
    """
    if total_chars == 0:
        return 100.0
    
    return (correct_chars / total_chars) * 100.0

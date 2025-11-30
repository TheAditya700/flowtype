from app.ml.difficulty import calculate_difficulty

def test_difficulty_calculation():
    """
    Tests the difficulty scoring logic.
    """
    # Simple case
    text1 = "the cat sat on the mat"
    features1 = calculate_difficulty(text1)
    assert features1['avg_word_length'] == 3.0
    assert features1['difficulty_score'] > 1.0

    # More complex case
    text2 = "Jaded, quizzical wizards vex nymphs."
    features2 = calculate_difficulty(text2)
    assert features2['avg_word_length'] > 5.0
    assert features2['punctuation_density'] > 0
    assert features2['rare_letter_count'] > 3
    assert features2['difficulty_score'] > features1['difficulty_score']

    # Empty case
    text3 = ""
    features3 = calculate_difficulty(text3)
    assert features3['difficulty_score'] == 1.0

from preprocessor import null_value_cleaner


def test_nnd():
    assert null_value_cleaner.nnd() == ('remove', [], []), 'Remove approach with no rows neither to keep nor remove'


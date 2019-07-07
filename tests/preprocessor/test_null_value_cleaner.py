from preprocessor import null_value_cleaner


def test_nnd():
    assert null_value_cleaner.nnd() == ('remove', [2], []), 'Remove approach with no rows neither to keep nor remove'


# if __name__ == '__main__':
#     no_parameter_nnd()
#     print("Everything passed")

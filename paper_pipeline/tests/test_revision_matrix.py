from paper_pipeline.scripts.revision_matrix import build_matrix, validate_matrix


def test_complete_matched_matrix_and_identical_profile_paths():
    matrix = build_matrix()
    validate_matrix(matrix)
    assert len(matrix['central']) == 6
    assert len(matrix['weather']) == 24
    assert len(matrix['siting']) == 375
    assert set(matrix) == {'central', 'weather', 'siting'}


def test_diagnostics_require_explicit_opt_in():
    matrix = build_matrix(include_diagnostics=True)
    assert len(matrix['sensitivity']) == 8
    assert len(matrix['full_year']) == 2
    assert len(matrix['numerics']) == 2
    central = next(c for c in matrix['central'] if c['scenario'] == 'all2030')
    for case in matrix['sensitivity'] + matrix['numerics']:
        if case['scenario'] == 'all2030':
            assert case['profile'] == central['profile']


def test_duplicate_output_is_rejected():
    import pytest
    matrix = build_matrix()
    matrix['weather'].append(matrix['central'][0])
    with pytest.raises(ValueError, match='duplicate'):
        validate_matrix(matrix)


def test_siting_keeps_2023_first_and_uses_per_year_seasonal_profiles():
    matrix = build_matrix()
    siting = matrix['siting']
    assert all(c['year'] == 2023 for c in siting[:75])
    assert sorted({c['year'] for c in siting[75:]}) == [2019, 2020, 2021, 2022]
    assert all(c['horizon'] == 'seasonal' for c in siting)
    for c in siting:
        if c['scenario'] == 'multiloc':
            assert c['profile'] == f"profiles/seasonal/{c['year']}/{c['location']}_{c['scale']}.csv"
    assert len({c['year'] for c in siting if c['scenario'] == 'none'}) == 5

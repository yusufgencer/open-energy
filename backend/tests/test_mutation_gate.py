import pytest

from openenergy.testing.mutation_gate import MutationScore, parse_results


def test_mutation_score_excludes_skipped_and_counts_timeouts_as_killed():
    score = parse_results(
        """
        openenergy.features.target.x__mutmut_1: killed
        openenergy.features.target.x__mutmut_2: timeout
        openenergy.features.target.x__mutmut_3: survived
        openenergy.features.target.x__mutmut_4: skipped
        """
    )

    assert score == MutationScore(killed=2, survived=1)
    assert score.percent == pytest.approx(200 / 3)


def test_empty_mutation_results_cannot_pass_the_gate():
    assert parse_results("").percent == 0.0

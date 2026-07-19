import numpy as np
import pytest
from openenergy.models.base import ModelWrapper, Prediction


def test_prediction_holds_quantiles():
    p = Prediction(p50=np.array([0.5]), p10=np.array([0.3]), p90=np.array([0.7]))
    assert p.p50[0] == 0.5 and p.p10[0] == 0.3


def test_modelwrapper_is_abstract():
    with pytest.raises(TypeError):
        ModelWrapper()

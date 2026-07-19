"""Per-prediction feature explanations using SHAP.

The model predicts log(SalePrice + 1), so SHAP values come out in log-space
too: they sum to (prediction_log - base_value), where base_value is the
model's average prediction over its training data. To show something a
person can actually read ("+$8,200 for OverallQual"), each SHAP value is
converted to an approximate dollar amount using a local linear
approximation -- the derivative of expm1 at the predicted point, exp(pred_log).
This is exact only for small contributions; for very large ones it's an
approximation, which is noted in the API docs.
"""

import numpy as np
import shap


def build_explainer(model):
    return shap.TreeExplainer(model)


def top_contributions(explainer, row, prediction_log, top_n=5):
    shap_values = explainer.shap_values(row)[0]
    dollar_scale = np.exp(prediction_log)

    contributions = [
        {"feature": feature, "impact_usd": float(value * dollar_scale)}
        for feature, value in zip(row.columns, shap_values)
    ]
    contributions.sort(key=lambda c: abs(c["impact_usd"]), reverse=True)
    return contributions[:top_n]

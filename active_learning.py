#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %pip install -U scikit-activeml gradio -q


# In[ ]:


import warnings

import gradio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import QueryByCommittee, RandomSampling, UncertaintySampling
from skactiveml.utils import MISSING_LABEL
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


# # Load datasets

# In[ ]:


iris = load_iris()
digits = load_digits()


# # Model Before Active Learning

# In[ ]:


Model_before_active_learning = RandomForestClassifier(n_estimators=2, random_state=42)


# ## Iris

# In[ ]:


Model_before_active_learning.fit(iris.data, iris.target)
Model_before_active_learning.score(iris.data, iris.target), f1_score(iris.target, Model_before_active_learning.predict(iris.data), average="micro")


# ## Digits

# In[ ]:


Model_before_active_learning.fit(digits.data, digits.target)
Model_before_active_learning.score(digits.data, digits.target), f1_score(digits.target, Model_before_active_learning.predict(digits.data), average="micro")


# # Model After Active Learning

# ## Active Learning Function

# In[ ]:


def evaluate_active_learning(dataset, dataset_name, method):
    X, y_true = dataset
    n_cycles = 100
    accuracies = []
    f1_scores = []
    qs = None

    print(f"X shape: {X.shape}, y_true shape: {y_true.shape}")
    print(f"Evaluating dataset: {dataset_name} with method: {method}")
    global random_forest
    random_forest = SklearnClassifier(
        RandomForestClassifier(), classes=np.unique(y_true)
    )
    if method == 0:
        qs = RandomSampling(random_state=42)
    elif method == 1:
        qs = QueryByCommittee(random_state=42, method="KL_divergence")
    elif method == 2:
        qs = UncertaintySampling(method="least_confident", random_state=42)
    elif method == 3:
        qs = UncertaintySampling(method="margin_sampling", random_state=42)
    elif method == 4:
        qs = UncertaintySampling(method="entropy", random_state=42)
    y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)
    random_forest.fit(X, y)
    y_hat = random_forest.predict(X)
    for _ in range(n_cycles):
        if method == 0:
            query_idx = qs.query(X=X, y=y, batch_size=1)
        elif method == 1:
            query_idx = qs.query(X=X, y=y, ensemble=random_forest, batch_size=1)
        else:
            query_idx = qs.query(X=X, y=y, clf=random_forest, batch_size=1)
        y[query_idx] = y_true[query_idx]
        random_forest.fit(X, y)
        y_hat = random_forest.predict(X)
        accuracy = random_forest.score(X, y_true)
        f1 = f1_score(y_true, y_hat, average="micro")
        f1_scores.append(f1)
        accuracies.append(accuracy)
    return accuracies, n_cycles, f1_scores, random_forest


# ## Predict Function to be used at "Try Model" Page in GUI

# In[ ]:


def model_predict(prompt):
    return str(random_forest.predict(prompt))


# ## Plotting Function to be used at "Active Learning" Page in GUI

# In[ ]:


def get_plot(methods, dataset_name):
    print(methods, dataset_name)
    dataset = {"Iris": [iris.data, iris.target], "Digits": [digits.data, digits.target]}
    df1s = []
    df2s = []
    for method in methods:
        if method == "Random Sampling":
            method = 0
        elif method == "Query By Committee":
            method = 1
        elif method == "Uncertainty Sampling with Least Confident":
            method = 2
        elif method == "Uncertainty Sampling with Margin Sampling":
            method = 3
        elif method == "Uncertainty Sampling with Entropy":
            method = 4
        accuracies, n, f1, _ = evaluate_active_learning(
            dataset[dataset_name], dataset_name, method
        )
        df1 = pd.DataFrame({"x": range(1, n + 1), "y": accuracies})
        df2 = pd.DataFrame({"x": range(1, n + 1), "y": f1})
        df1s.append(df1)
        df2s.append(df2)
    fig1 = make_subplots(
        rows=1,
        cols=2,
    )
    fig2 = make_subplots(
        rows=1,
        cols=2,
    )
    for df1 in df1s:
        fig1.add_trace(go.Scatter(x=df1["x"], y=df1["y"], mode="lines"), row=1, col=1)
    for df2 in df2s:
        fig2.add_trace(go.Scatter(x=df2["x"], y=df2["y"], mode="lines"), row=1, col=1)
    fig1.update_layout(title_text="Accuracy")
    fig2.update_layout(title_text="F1 Score")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
    for trace in fig1["data"]:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2["data"]:
        fig.add_trace(trace, row=1, col=2)
    fig.update_layout(title_text="Accuracy vs F1 Score")
    fig.update_xaxes(title_text="Iterations")
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="F1 Score", row=1, col=2)
    fig.update_layout(height=500, width=500)
    return fig


# ## "Try Model" Page in GUI

# In[ ]:


try_model = gradio.Interface(
    fn=model_predict,
    inputs=gradio.inputs.Dataframe(
        type="numpy",
        row_count=1,
        col_count=4,
        headers=["sepal length", "sepal width", "petal length", "petal width"],
    ),
    outputs="text",
    title="Active Learning with Random Forest",
    description="This app apply active learning using a Random Forest classifier.",
)


# ## "Active Learning" Page in GUI

# In[ ]:


with gradio.Blocks() as main:
    with gradio.Row():
        with gradio.Column():
            methods = gradio.CheckboxGroup(
                [
                    "Random Sampling",
                    "Query By Committee",
                    "Uncertainty Sampling with Least Confident",
                    "Uncertainty Sampling with Margin Sampling",
                    "Uncertainty Sampling with Entropy",
                ],
                label="Select Active Learning Method",
            )
            dataset = gradio.Radio(["Iris", "Digits"], label="Select Dataset")
            with gradio.Row():
                with gradio.Column():
                    output = gradio.Plot(
                        x="x",
                        y="y",
                        overlay_point=True,
                        tooltip=["x", "y"],
                        y_title="Accuracy",
                        x_title="Iterations",
                    )
                    submit = gradio.Button()
    submit.click(get_plot, [methods, dataset], output)


# ## GUI

# In[ ]:


gradio.TabbedInterface([main, try_model], ["Active Learning", "Try Model"]).launch()


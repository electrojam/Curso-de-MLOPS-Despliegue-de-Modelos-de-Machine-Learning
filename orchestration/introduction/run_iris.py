from prefect import task, flow
from datetime import timedelta
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


@task(
    name = "Load Iris Dataset",
    tags = ["data_loading"],
    description = "Load Iris dataset from sklearn"
)

def get_data_from_sklearn() -> dict:
    """This function loas the iris dataset from sklearn"""
    data = load_iris()
    return {"data": data.data, "target": data.target}

@flow(retries=3, retry_delay_seconds=5, log_prints=True)
def iris_classification():
    """This function orchestrates the whole flow"""
    data= get_data_from_sklearn()
    print(data)

iris_classification()
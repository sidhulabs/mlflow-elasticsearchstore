from datetime import datetime

from mlflow.entities import (
    Experiment,
    ExperimentTag,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunTag,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from pydantic import BaseModel


class ElasticExperiment(BaseModel):
    experiment_id: str
    name: str
    artifact_location: str
    tags: ExperimentTag
    lifecycle_stage: str = LifecycleStage.ACTIVE
    creation_time: datetime = datetime.now()
    last_update_time: datetime = datetime.now()

    def to_mlflow_entity(self) -> Experiment:
        return Experiment(
            experiment_id=self.experiment_id,
            name=self.name,
            artifact_location=self.artifact_location,
            lifecycle_stage=self.lifecycle_stage,
            tags=[t.to_mlflow_entity() for t in self.tags],
            creation_time=self.creation_time,
            last_update_time=self.last_update_time,
        )


class ElasticExperimentTag(BaseModel):
    key: str
    value: str
    experiment_id: str

    def to_mlflow_entity(self) -> ExperimentTag:
        return ExperimentTag(key=self.key, value=self.value)


class ElasticTag(BaseModel):
    key: str
    value: str
    run_id: str

    def to_mlflow_entity(self) -> RunTag:
        return RunTag(key=self.key, value=self.value)


class ElasticMetric(BaseModel):
    key: str
    value: float
    timestamp: datetime = datetime.now()
    step: int
    is_nan: bool
    run_id: str

    def to_mlflow_entity(self) -> Metric:
        return Metric(
            key=self.key,
            value=float("nan") if self.is_nan else self.value,
            timestamp=self.timestamp,
            step=self.step,
        )


class ElasticLatestMetric(BaseModel):
    key: str
    value: float
    timestamp: datetime = datetime.now()
    step: int
    is_nan: bool

    def to_mlflow_entity(self) -> Metric:
        return Metric(
            key=self.key,
            value=float("nan") if self.is_nan else self.value,
            timestamp=self.timestamp,
            step=self.step,
        )


class ElasticParam(BaseModel):
    key: str
    value: str
    run_id: str

    def to_mlflow_entity(self) -> Param:
        return Param(key=self.key, value=self.value)


class ElasticRun(BaseModel):
    run_id: str
    name: str
    source_type: str
    source_name: str
    experiment_id: str
    user_id: str
    status: str
    start_time: datetime = datetime.now()
    end_time: datetime
    source_version: str
    lifecycle_stage: str = LifecycleStage.ACTIVE
    artifact_uri: str
    latest_metrics: ElasticLatestMetric
    params: ElasticParam
    tags: ElasticTag

    def to_mlflow_entity(self) -> Run:
        run_info = RunInfo(
            run_uuid=self.run_id,
            run_id=self.run_id,
            experiment_id=str(self.experiment_id),
            user_id=self.user_id,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            lifecycle_stage=self.lifecycle_stage,
            artifact_uri=self.artifact_uri,
        )

        run_data = RunData(
            metrics=[m.to_mlflow_entity() for m in self.latest_metrics],
            params=[p.to_mlflow_entity() for p in self.params],
            tags=[t.to_mlflow_entity() for t in self.tags],
        )
        return Run(run_info=run_info, run_data=run_data)

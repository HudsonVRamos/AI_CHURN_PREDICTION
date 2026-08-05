"""Pipeline de Machine Learning no Amazon SageMaker.

Responsável por:
- Treinar modelos supervisionados (XGBoost, LightGBM, CatBoost)
- Executar inferência em lote via Batch Transform
- Registrar modelos no SageMaker Model Registry
- Computar métricas de avaliação (Precision, Recall, F1, ROC AUC)
- Split estratificado: train 70%, validation 15%, test 15%

Requirements: 10.1, 10.2, 10.3, 10.7, 10.8, 10.10
"""

from __future__ import annotations

import io
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.common.logging import get_logger
from src.common.models import ModelVersion

logger = get_logger("ml-inference")


# Mapeamento de algoritmos para imagens SageMaker
ALGORITHM_IMAGE_MAP = {
    "xgboost": "{region}.dkr.ecr.{region}.amazonaws.com/xgboost:latest",
    "lightgbm": "{region}.dkr.ecr.{region}.amazonaws.com/lightgbm:latest",
    "catboost": "{account}.dkr.ecr.{region}.amazonaws.com/catboost-custom:latest",
}

# Hyperparameters padrão por algoritmo
DEFAULT_HYPERPARAMETERS: dict[str, dict[str, str]] = {
    "xgboost": {
        "objective": "binary:logistic",
        "num_round": "100",
        "max_depth": "6",
        "eta": "0.3",
        "eval_metric": "auc",
        "seed": "42",
    },
    "lightgbm": {
        "objective": "binary",
        "num_iterations": "100",
        "max_depth": "6",
        "learning_rate": "0.3",
        "metric": "auc",
        "seed": "42",
    },
    "catboost": {
        "loss_function": "Logloss",
        "iterations": "100",
        "depth": "6",
        "learning_rate": "0.3",
        "eval_metric": "AUC",
        "random_seed": "42",
    },
}


class SageMakerMLPipeline:
    """Pipeline de treino e inferência no Amazon SageMaker.

    Suporta XGBoost (built-in), LightGBM (built-in) e CatBoost
    (custom container). O algoritmo ativo é configurável via config.

    Attributes:
        SUPPORTED_ALGORITHMS: Lista de algoritmos suportados.
    """

    SUPPORTED_ALGORITHMS = ["xgboost", "lightgbm", "catboost"]

    def __init__(
        self,
        region: str,
        model_package_group: str,
        role_arn: str,
        bucket: str,
        algorithm: str = "xgboost",
        training_instance: str = "ml.m5.xlarge",
        batch_instance: str = "ml.m5.large",
        sagemaker_client: Any | None = None,
        s3_client: Any | None = None,
    ) -> None:
        """Inicializa o pipeline SageMaker.

        Args:
            region: Região AWS para os serviços.
            model_package_group: Nome do grupo no Model Registry.
            role_arn: ARN do IAM Role para execução no SageMaker.
            bucket: Bucket S3 para armazenamento de dados e artefatos.
            algorithm: Algoritmo padrão (xgboost|lightgbm|catboost).
            training_instance: Tipo de instância para treino.
            batch_instance: Tipo de instância para batch transform.
            sagemaker_client: Cliente boto3 SageMaker (para testes).
            s3_client: Cliente boto3 S3 (para testes).

        Raises:
            ValueError: Se o algoritmo não é suportado.
        """
        if algorithm.lower() not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Algoritmo '{algorithm}' não suportado. "
                f"Use: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

        self.region = region
        self.model_package_group = model_package_group
        self.role_arn = role_arn
        self.bucket = bucket
        self.algorithm = algorithm.lower()
        self.training_instance = training_instance
        self.batch_instance = batch_instance

        self._sagemaker_client = sagemaker_client or boto3.client(
            "sagemaker", region_name=region
        )
        self._s3_client = s3_client or boto3.client(
            "s3", region_name=region
        )

        logger.info(
            f"SageMakerMLPipeline inicializado: "
            f"region={region}, algorithm={self.algorithm}, "
            f"model_package_group={model_package_group}"
        )

    def train(
        self,
        training_data_s3: str,
        algorithm: str | None = None,
        hyperparameters: dict | None = None,
    ) -> ModelVersion:
        """Treina modelo no SageMaker e registra no Model Registry.

        Etapas:
        1. Download do dataset do S3
        2. Split estratificado (70/15/15)
        3. Upload dos splits para S3
        4. Criação do Training Job no SageMaker
        5. Avaliação no test set (Precision, Recall, F1, ROC AUC)
        6. Registro no Model Registry com métricas

        Args:
            training_data_s3: Path S3 do dataset (CSV com features + label).
            algorithm: Algoritmo a usar (override do padrão da instância).
            hyperparameters: Hyperparameters customizados (override dos defaults).

        Returns:
            ModelVersion com metadados do modelo registrado.

        Raises:
            ValueError: Se algoritmo não é suportado.
            RuntimeError: Se o Training Job falhar.
        """
        algo = (algorithm or self.algorithm).lower()
        if algo not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Algoritmo '{algo}' não suportado. "
                f"Use: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

        logger.log_stage_start()
        start_time = time.time()
        model_version_id = str(uuid.uuid4())[:8]
        training_date = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Iniciando treino: algorithm={algo}, "
            f"data={training_data_s3}, version={model_version_id}"
        )

        # 1. Download e preparação dos dados
        df = self._download_training_data(training_data_s3)

        # 2. Split estratificado: train 70%, validation 15%, test 15%
        train_df, val_df, test_df = self._stratified_split(df)

        # 3. Upload dos splits para S3
        s3_prefix = f"models/{model_version_id}"
        train_s3 = self._upload_split(train_df, s3_prefix, "train")
        val_s3 = self._upload_split(val_df, s3_prefix, "validation")
        test_s3 = self._upload_split(test_df, s3_prefix, "test")

        # 4. Criar Training Job
        job_name = f"churn-{algo}-{model_version_id}"
        final_hyperparams = self._get_hyperparameters(algo, hyperparameters)
        model_artifact_s3 = self._create_training_job(
            job_name=job_name,
            algorithm=algo,
            hyperparameters=final_hyperparams,
            train_s3=train_s3,
            validation_s3=val_s3,
            output_s3=f"s3://{self.bucket}/{s3_prefix}/output",
        )

        # 5. Avaliar no test set
        metrics = self._evaluate_model(test_df, model_artifact_s3, algo)

        logger.info(
            f"Métricas do modelo: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

        # 6. Registrar no Model Registry
        model_package_arn = self._register_model(
            model_artifact_s3=model_artifact_s3,
            algorithm=algo,
            metrics=metrics,
            hyperparameters=final_hyperparams,
            model_version_id=model_version_id,
        )

        # Upload de métricas e hyperparameters para S3
        self._upload_metadata(
            s3_prefix=s3_prefix,
            metrics=metrics,
            hyperparameters=final_hyperparams,
        )

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)

        return ModelVersion(
            model_package_arn=model_package_arn,
            algorithm=algo,
            training_date=training_date,
            dataset_version=training_data_s3,
            metrics=metrics,
        )

    def predict_batch(
        self,
        feature_vectors_s3: str,
        model_version: str | None = None,
    ) -> str:
        """Executa Batch Transform e retorna S3 path dos resultados.

        Args:
            feature_vectors_s3: Path S3 dos features para inferência.
            model_version: ARN do model package. Se None, usa modelo aprovado.

        Returns:
            S3 path do diretório com resultados do batch transform.

        Raises:
            RuntimeError: Se o Batch Transform falhar.
        """
        logger.info(
            f"Iniciando Batch Transform: input={feature_vectors_s3}, "
            f"model_version={model_version or 'approved'}"
        )

        if model_version is None:
            active_model = self.get_active_model()
            model_version = active_model.model_package_arn

        transform_job_name = f"churn-batch-{str(uuid.uuid4())[:8]}"
        output_s3 = (
            f"s3://{self.bucket}/predictions/"
            f"{transform_job_name}"
        )

        self._create_batch_transform_job(
            job_name=transform_job_name,
            model_package_arn=model_version,
            input_s3=feature_vectors_s3,
            output_s3=output_s3,
        )

        logger.info(
            f"Batch Transform concluído: output={output_s3}"
        )
        return output_s3

    def get_active_model(self) -> ModelVersion:
        """Retorna o modelo atualmente aprovado no Model Registry.

        Consulta o SageMaker Model Registry pelo modelo com status
        'Approved' mais recente no grupo configurado.

        Returns:
            ModelVersion com metadados do modelo ativo.

        Raises:
            RuntimeError: Se nenhum modelo aprovado encontrado.
        """
        logger.info(
            f"Buscando modelo aprovado no grupo: "
            f"{self.model_package_group}"
        )

        response = self._sagemaker_client.list_model_packages(
            ModelPackageGroupName=self.model_package_group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )

        packages = response.get("ModelPackageSummaryList", [])
        if not packages:
            raise RuntimeError(
                f"Nenhum modelo aprovado encontrado no grupo "
                f"'{self.model_package_group}'"
            )

        package_arn = packages[0]["ModelPackageArn"]
        package_detail = self._sagemaker_client.describe_model_package(
            ModelPackageName=package_arn
        )

        # Extrair métricas do model package via CustomerMetadataProperties
        # O SageMaker retorna CustomerMetadataProperties como dict simples
        custom_metadata = package_detail.get(
            "CustomerMetadataProperties", {}
        )
        if not isinstance(custom_metadata, dict):
            custom_metadata = {}

        model_metrics = {
            "precision": float(custom_metadata.get("precision", 0.0)),
            "recall": float(custom_metadata.get("recall", 0.0)),
            "f1": float(custom_metadata.get("f1", 0.0)),
            "roc_auc": float(custom_metadata.get("roc_auc", 0.0)),
        }

        algorithm = custom_metadata.get("algorithm", self.algorithm)
        training_date = custom_metadata.get(
            "training_date",
            packages[0].get("CreationTime", "").isoformat()
            if hasattr(packages[0].get("CreationTime", ""), "isoformat")
            else str(packages[0].get("CreationTime", "")),
        )
        dataset_version = custom_metadata.get("dataset_version", "unknown")

        return ModelVersion(
            model_package_arn=package_arn,
            algorithm=algorithm,
            training_date=training_date,
            dataset_version=dataset_version,
            metrics=model_metrics,
        )

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _download_training_data(self, s3_path: str) -> pd.DataFrame:
        """Download do dataset CSV do S3 e retorna como DataFrame.

        Espera CSV com coluna 'label' (0=active, 1=churned) e demais
        colunas como features numéricas.

        Args:
            s3_path: Path S3 no formato s3://bucket/key.

        Returns:
            DataFrame com os dados de treinamento.
        """
        bucket, key = self._parse_s3_path(s3_path)
        logger.info(f"Download training data: s3://{bucket}/{key}")

        response = self._s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response["Body"].read().decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_content))

        logger.info(
            f"Dataset carregado: {len(df)} registros, "
            f"{len(df.columns)} colunas"
        )
        return df

    def _stratified_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split estratificado: train 70%, validation 15%, test 15%.

        Utiliza sklearn.train_test_split com stratify para manter
        a proporção de classes (label=0, label=1) em cada split.

        Args:
            df: DataFrame com coluna 'label' e features.

        Returns:
            Tupla (train_df, val_df, test_df).
        """
        labels = df["label"]

        # Primeiro split: 70% train, 30% restante
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=42,
            stratify=labels,
        )

        # Segundo split: 50% do restante para val (15%), 50% para test (15%)
        temp_labels = temp_df["label"]
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=42,
            stratify=temp_labels,
        )

        logger.info(
            f"Split estratificado: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df

    def _upload_split(
        self, df: pd.DataFrame, s3_prefix: str, split_name: str
    ) -> str:
        """Upload de um split do dataset para S3 em formato CSV.

        Para algoritmos built-in do SageMaker, o CSV não deve conter
        header e a primeira coluna deve ser o label.

        Args:
            df: DataFrame do split.
            s3_prefix: Prefixo S3 para armazenamento.
            split_name: Nome do split (train/validation/test).

        Returns:
            S3 URI do arquivo uploaded.
        """
        # SageMaker built-in espera: label na primeira coluna, sem header
        cols = ["label"] + [c for c in df.columns if c != "label"]
        csv_buffer = io.StringIO()
        df[cols].to_csv(csv_buffer, index=False, header=False)

        key = f"{s3_prefix}/{split_name}/data.csv"
        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=csv_buffer.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )

        s3_uri = f"s3://{self.bucket}/{key}"
        logger.info(f"Upload {split_name}: {s3_uri} ({len(df)} registros)")
        return s3_uri

    def _get_hyperparameters(
        self, algorithm: str, overrides: dict | None
    ) -> dict[str, str]:
        """Retorna hyperparameters mesclando defaults com overrides.

        Args:
            algorithm: Algoritmo selecionado.
            overrides: Hyperparameters customizados (opcional).

        Returns:
            Dict com hyperparameters finais (todos como string).
        """
        params = dict(DEFAULT_HYPERPARAMETERS.get(algorithm, {}))
        if overrides:
            params.update({k: str(v) for k, v in overrides.items()})
        return params

    def _get_training_image(self, algorithm: str) -> str:
        """Retorna a URI da imagem Docker para o algoritmo.

        Args:
            algorithm: Algoritmo selecionado.

        Returns:
            URI da imagem ECR.
        """
        # Obter account ID para custom containers
        sts = boto3.client("sts", region_name=self.region)
        try:
            account_id = sts.get_caller_identity()["Account"]
        except Exception:
            account_id = "000000000000"

        template = ALGORITHM_IMAGE_MAP[algorithm]
        return template.format(region=self.region, account=account_id)

    def _create_training_job(
        self,
        job_name: str,
        algorithm: str,
        hyperparameters: dict[str, str],
        train_s3: str,
        validation_s3: str,
        output_s3: str,
    ) -> str:
        """Cria e aguarda conclusão de um Training Job no SageMaker.

        Args:
            job_name: Nome único do training job.
            algorithm: Algoritmo selecionado.
            hyperparameters: Hyperparameters do modelo.
            train_s3: S3 URI dos dados de treino.
            validation_s3: S3 URI dos dados de validação.
            output_s3: S3 URI para output do modelo.

        Returns:
            S3 URI do artefato do modelo treinado.

        Raises:
            RuntimeError: Se o training job falhar.
        """
        training_image = self._get_training_image(algorithm)

        self._sagemaker_client.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                "TrainingImage": training_image,
                "TrainingInputMode": "File",
            },
            RoleArn=self.role_arn,
            HyperParameters=hyperparameters,
            InputDataConfig=[
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": train_s3,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "text/csv",
                },
                {
                    "ChannelName": "validation",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": validation_s3,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "text/csv",
                },
            ],
            OutputDataConfig={"S3OutputPath": output_s3},
            ResourceConfig={
                "InstanceType": self.training_instance,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
        )

        logger.info(f"Training job criado: {job_name}")

        # Aguardar conclusão do training job
        model_artifact_s3 = self._wait_for_training_job(job_name, output_s3)
        return model_artifact_s3

    def _wait_for_training_job(
        self, job_name: str, output_s3: str
    ) -> str:
        """Aguarda conclusão do training job via polling.

        Args:
            job_name: Nome do training job.
            output_s3: S3 path base do output.

        Returns:
            S3 URI do artefato do modelo.

        Raises:
            RuntimeError: Se o job falhar ou for interrompido.
        """
        while True:
            response = self._sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )
            status = response["TrainingJobStatus"]

            if status == "Completed":
                model_artifact = response.get(
                    "ModelArtifacts", {}
                ).get(
                    "S3ModelArtifacts",
                    f"{output_s3}/{job_name}/output/model.tar.gz",
                )
                logger.info(
                    f"Training job concluído: {job_name}, "
                    f"artifact={model_artifact}"
                )
                return model_artifact

            if status in ("Failed", "Stopped"):
                failure_reason = response.get(
                    "FailureReason", "Motivo não disponível"
                )
                raise RuntimeError(
                    f"Training job '{job_name}' falhou: {failure_reason}"
                )

            logger.debug(f"Training job {job_name}: status={status}")
            time.sleep(30)

    def _evaluate_model(
        self,
        test_df: pd.DataFrame,
        model_artifact_s3: str,
        algorithm: str,
    ) -> dict[str, float]:
        """Avalia o modelo no test set e computa métricas.

        Computa Precision, Recall, F1-Score e ROC AUC no test set.
        Para avaliação, utiliza o SageMaker Batch Transform no test set
        ou avaliação local quando possível.

        Args:
            test_df: DataFrame do test set com label e features.
            model_artifact_s3: S3 URI do modelo treinado.
            algorithm: Algoritmo utilizado.

        Returns:
            Dict com métricas: precision, recall, f1, roc_auc.
        """
        # Upload do test set para batch transform de avaliação
        eval_id = str(uuid.uuid4())[:8]
        test_features = test_df.drop(columns=["label"])
        test_labels = test_df["label"].values

        # Upload features do test set (sem label) para inferência
        csv_buffer = io.StringIO()
        test_features.to_csv(csv_buffer, index=False, header=False)

        eval_key = f"evaluation/{eval_id}/test_features.csv"
        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=eval_key,
            Body=csv_buffer.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )

        # Executar batch transform para obter predições no test set
        eval_input_s3 = f"s3://{self.bucket}/{eval_key}"
        eval_output_s3 = f"s3://{self.bucket}/evaluation/{eval_id}/output"
        eval_job_name = f"churn-eval-{eval_id}"

        self._create_batch_transform_job(
            job_name=eval_job_name,
            model_package_arn=model_artifact_s3,
            input_s3=eval_input_s3,
            output_s3=eval_output_s3,
            is_evaluation=True,
        )

        # Download das predições
        predictions = self._download_predictions(
            f"{eval_output_s3}/test_features.csv.out"
        )

        # Calcular métricas
        if len(predictions) != len(test_labels):
            logger.warning(
                f"Tamanho das predições ({len(predictions)}) difere do "
                f"test set ({len(test_labels)}). Usando mínimo."
            )
            min_len = min(len(predictions), len(test_labels))
            predictions = predictions[:min_len]
            test_labels = test_labels[:min_len]

        # Converter probabilidades em classes (threshold=0.5)
        predicted_classes = (
            np.array(predictions) >= 0.5
        ).astype(int)

        metrics = {
            "precision": float(precision_score(
                test_labels, predicted_classes, zero_division=0
            )),
            "recall": float(recall_score(
                test_labels, predicted_classes, zero_division=0
            )),
            "f1": float(f1_score(
                test_labels, predicted_classes, zero_division=0
            )),
            "roc_auc": float(roc_auc_score(
                test_labels, predictions
            )),
        }

        return metrics

    def _download_predictions(self, s3_path: str) -> list[float]:
        """Download das predições do batch transform.

        Args:
            s3_path: S3 URI do arquivo de predições.

        Returns:
            Lista de probabilidades de churn.
        """
        bucket, key = self._parse_s3_path(s3_path)
        response = self._s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")

        predictions = []
        for line in content.strip().split("\n"):
            if line.strip():
                predictions.append(float(line.strip()))

        return predictions

    def _create_batch_transform_job(
        self,
        job_name: str,
        model_package_arn: str,
        input_s3: str,
        output_s3: str,
        is_evaluation: bool = False,
    ) -> None:
        """Cria e aguarda conclusão de um Batch Transform Job.

        Args:
            job_name: Nome único do transform job.
            model_package_arn: ARN do model package ou S3 do artefato.
            input_s3: S3 URI dos dados de entrada.
            output_s3: S3 URI para os resultados.
            is_evaluation: Se True, usa modelo por artefato S3.

        Raises:
            RuntimeError: Se o batch transform falhar.
        """
        # Criar modelo SageMaker para o transform
        model_name = f"model-{job_name}"

        if is_evaluation:
            # Usar artefato S3 diretamente
            training_image = self._get_training_image(self.algorithm)
            self._sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    "Image": training_image,
                    "ModelDataUrl": model_package_arn,
                },
                ExecutionRoleArn=self.role_arn,
            )
        else:
            # Usar Model Package do Registry
            self._sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    "ModelPackageName": model_package_arn,
                },
                ExecutionRoleArn=self.role_arn,
            )

        self._sagemaker_client.create_transform_job(
            TransformJobName=job_name,
            ModelName=model_name,
            TransformInput={
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_s3,
                    }
                },
                "ContentType": "text/csv",
                "SplitType": "Line",
            },
            TransformOutput={
                "S3OutputPath": output_s3,
                "AssembleWith": "Line",
            },
            TransformResources={
                "InstanceType": self.batch_instance,
                "InstanceCount": 1,
            },
        )

        logger.info(f"Batch Transform job criado: {job_name}")
        self._wait_for_transform_job(job_name)

    def _wait_for_transform_job(self, job_name: str) -> None:
        """Aguarda conclusão do batch transform job via polling.

        Args:
            job_name: Nome do transform job.

        Raises:
            RuntimeError: Se o job falhar ou for interrompido.
        """
        while True:
            response = self._sagemaker_client.describe_transform_job(
                TransformJobName=job_name
            )
            status = response["TransformJobStatus"]

            if status == "Completed":
                logger.info(f"Transform job concluído: {job_name}")
                return

            if status in ("Failed", "Stopped"):
                failure_reason = response.get(
                    "FailureReason", "Motivo não disponível"
                )
                raise RuntimeError(
                    f"Transform job '{job_name}' falhou: {failure_reason}"
                )

            logger.debug(f"Transform job {job_name}: status={status}")
            time.sleep(30)

    def _register_model(
        self,
        model_artifact_s3: str,
        algorithm: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, str],
        model_version_id: str,
    ) -> str:
        """Registra modelo no SageMaker Model Registry.

        Args:
            model_artifact_s3: S3 URI do artefato do modelo.
            algorithm: Algoritmo utilizado.
            metrics: Métricas de avaliação.
            hyperparameters: Hyperparameters usados no treino.
            model_version_id: ID de versão do modelo.

        Returns:
            ARN do Model Package registrado.
        """
        training_image = self._get_training_image(algorithm)
        training_date = datetime.now(timezone.utc).isoformat()

        response = self._sagemaker_client.create_model_package(
            ModelPackageGroupName=self.model_package_group,
            ModelPackageDescription=(
                f"Churn prediction model - {algorithm} - "
                f"v{model_version_id}"
            ),
            InferenceSpecification={
                "Containers": [
                    {
                        "Image": training_image,
                        "ModelDataUrl": model_artifact_s3,
                    }
                ],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
                "SupportedTransformInstanceTypes": [
                    self.batch_instance
                ],
                "SupportedRealtimeInferenceInstanceTypes": [
                    self.batch_instance
                ],
            },
            ModelApprovalStatus="PendingManualApproval",
            CustomerMetadataProperties={
                "algorithm": algorithm,
                "training_date": training_date,
                "dataset_version": model_version_id,
                "precision": str(metrics["precision"]),
                "recall": str(metrics["recall"]),
                "f1": str(metrics["f1"]),
                "roc_auc": str(metrics["roc_auc"]),
            },
        )

        model_package_arn = response["ModelPackageArn"]
        logger.info(
            f"Modelo registrado no Registry: {model_package_arn}"
        )
        return model_package_arn

    def _upload_metadata(
        self,
        s3_prefix: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, str],
    ) -> None:
        """Upload de métricas e hyperparameters para S3.

        Armazena artefatos de treino no path com versão do modelo
        conforme R10.10.

        Args:
            s3_prefix: Prefixo S3 (inclui versão do modelo).
            metrics: Métricas de avaliação.
            hyperparameters: Hyperparameters utilizados.
        """
        import json

        # Upload métricas
        metrics_key = f"{s3_prefix}/metrics.json"
        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=metrics_key,
            Body=json.dumps(metrics, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        # Upload hyperparameters
        hp_key = f"{s3_prefix}/hyperparameters.json"
        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=hp_key,
            Body=json.dumps(hyperparameters, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        logger.info(
            f"Metadados uploaded: s3://{self.bucket}/{s3_prefix}/"
        )

    @staticmethod
    def _parse_s3_path(s3_path: str) -> tuple[str, str]:
        """Parseia um S3 URI em bucket e key.

        Args:
            s3_path: Path no formato s3://bucket/key/path.

        Returns:
            Tupla (bucket, key).

        Raises:
            ValueError: Se o path não é um S3 URI válido.
        """
        if not s3_path.startswith("s3://"):
            raise ValueError(
                f"Path S3 inválido (deve iniciar com s3://): {s3_path}"
            )
        path_without_prefix = s3_path[5:]
        parts = path_without_prefix.split("/", 1)
        if len(parts) < 2:
            raise ValueError(
                f"Path S3 inválido (sem key): {s3_path}"
            )
        return parts[0], parts[1]

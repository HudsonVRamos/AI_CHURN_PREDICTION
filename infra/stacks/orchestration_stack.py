"""
Stack CDK para orquestração do pipeline de churn prediction.

Provisiona:
- Lambda functions para cada handler do pipeline
- Step Functions state machine (pipeline completo)
- EventBridge rule com agendamento semanal (cron)
- S3 event notification para trigger via upload

Requirements: 8.1, 14.6, 16.1, 17.3
"""

from aws_cdk import (
    Duration,
    Stack,
    aws_dynamodb as dynamodb,
    aws_events as events,
    aws_events_targets as events_targets,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_stepfunctions as sfn,
)
from constructs import Construct


class OrchestrationStack(Stack):
    """Stack de orquestração: Step Functions, Lambdas, EventBridge."""

    # Handlers do pipeline e seus nomes de função
    HANDLER_NAMES = [
        "ingest",
        "extract",
        "feature",
        "store",
        "predict",
        "shap",
        "bedrock",
        "report",
    ]

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        bucket: s3.IBucket,
        lambda_role: iam.IRole,
        feature_store_table: dynamodb.ITable,
        predictions_table: dynamodb.ITable,
        executions_table: dynamodb.ITable,
        **kwargs,
    ) -> None:
        """Inicializa o stack de orquestração.

        Args:
            bucket: Bucket S3 para dados do pipeline.
            lambda_role: IAM Role para execução dos Lambdas.
            feature_store_table: Tabela DynamoDB do Feature Store.
            predictions_table: Tabela DynamoDB de predições.
            executions_table: Tabela DynamoDB de execuções.
        """
        super().__init__(scope, construct_id, **kwargs)

        # Referências externas
        self._bucket = bucket
        self._lambda_role = lambda_role
        self._feature_store_table = feature_store_table
        self._predictions_table = predictions_table
        self._executions_table = executions_table

        # Criar Lambda functions para cada handler
        self.lambda_functions = self._create_lambda_functions()

        # Criar Step Functions state machine
        self.state_machine = self._create_state_machine()

        # EventBridge rule — agendamento semanal (segunda às 08:00 UTC)
        self.weekly_rule = self._create_weekly_schedule()

        # S3 event trigger — upload no prefixo input/
        self._create_s3_trigger()

    def _create_lambda_functions(self) -> dict[str, _lambda.Function]:
        """Cria Lambda functions para cada handler do pipeline."""
        functions: dict[str, _lambda.Function] = {}

        # Variáveis de ambiente comuns a todos os handlers
        common_env = {
            "BUCKET_NAME": self._bucket.bucket_name,
            "FEATURE_STORE_TABLE": self._feature_store_table.table_name,
            "PREDICTIONS_TABLE": self._predictions_table.table_name,
            "EXECUTIONS_TABLE": self._executions_table.table_name,
            "LOG_LEVEL": "INFO",
        }

        # Configuração por handler (timeout e memória ajustados)
        handler_config = {
            "ingest": {
                "timeout": Duration.seconds(60),
                "memory": 256,
                "description": "Ingestão e validação de listas de user IDs",
            },
            "extract": {
                "timeout": Duration.minutes(15),
                "memory": 512,
                "description": "Extração de dados da API NPAW",
            },
            "feature": {
                "timeout": Duration.minutes(10),
                "memory": 1024,
                "description": "Engenharia de features comportamentais",
            },
            "store": {
                "timeout": Duration.minutes(5),
                "memory": 256,
                "description": "Persistência no Feature Store (DynamoDB)",
            },
            "predict": {
                "timeout": Duration.minutes(15),
                "memory": 512,
                "description": "Inferência via SageMaker Batch Transform",
            },
            "shap": {
                "timeout": Duration.minutes(10),
                "memory": 2048,
                "description": "Cálculo de explicabilidade SHAP",
            },
            "bedrock": {
                "timeout": Duration.minutes(10),
                "memory": 512,
                "description": "Geração de explicações via AWS Bedrock",
            },
            "report": {
                "timeout": Duration.minutes(5),
                "memory": 512,
                "description": "Geração de relatórios JSON/Markdown",
            },
        }

        for handler_name in self.HANDLER_NAMES:
            config = handler_config[handler_name]
            fn = _lambda.Function(
                self,
                f"Handler-{handler_name}",
                function_name=f"churn-pipeline-{handler_name}",
                runtime=_lambda.Runtime.PYTHON_3_11,
                handler=(
                    f"src.orchestrator.handlers"
                    f".{handler_name}_handler.handler"
                ),
                code=_lambda.Code.from_asset(
                    "..",
                    exclude=[
                        "infra/*", "docs/*", "tests/*", ".git/*",
                        ".hypothesis/*", "*.pyc", "__pycache__/*",
                        ".venv/*", "node_modules/*", "cdk.out/*",
                    ],
                ),
                timeout=config["timeout"],
                memory_size=config["memory"],
                description=config["description"],
                role=self._lambda_role,
                environment=common_env,
            )
            functions[handler_name] = fn

        return functions

    def _create_state_machine(self) -> sfn.CfnStateMachine:
        """Cria a state machine do Step Functions com a definição do pipeline.

        Usa CfnStateMachine para injetar a definição ASL completa com
        os ARNs reais dos Lambdas criados.
        """
        import json
        import copy

        # Definição ASL inline (evita importar módulos de runtime pesados)
        definition = {
            "Comment": "Pipeline de predição de churn - Sky Brazil",
            "StartAt": "Ingestion",
            "States": {
                "Ingestion": {"Type": "Task", "Resource": "PLACEHOLDER", "Next": "Extraction",
                    "Retry": [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"], "IntervalSeconds": 5, "MaxAttempts": 3, "BackoffRate": 2.0}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "Extraction": {"Type": "Task", "Resource": "PLACEHOLDER", "Next": "FeatureEngineering",
                    "Retry": [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"], "IntervalSeconds": 5, "MaxAttempts": 3, "BackoffRate": 2.0}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "FeatureEngineering": {"Type": "Task", "Resource": "PLACEHOLDER", "Next": "StoreFeatures",
                    "Retry": [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"], "IntervalSeconds": 5, "MaxAttempts": 3, "BackoffRate": 2.0}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "StoreFeatures": {"Type": "Task", "Resource": "PLACEHOLDER", "Next": "ChooseMode",
                    "Retry": [{"ErrorEquals": ["States.TaskFailed", "DynamoDB.ProvisionedThroughputExceededException"], "IntervalSeconds": 2, "MaxAttempts": 5, "BackoffRate": 2.0}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "ChooseMode": {"Type": "Choice", "Choices": [
                    {"Variable": "$.mode", "StringEquals": "train", "Next": "Training"},
                    {"Variable": "$.mode", "StringEquals": "predict", "Next": "BatchPredict"}
                ], "Default": "BatchPredict"},
                "Training": {"Type": "Task", "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync", "Next": "EvaluateModel",
                    "Retry": [{"ErrorEquals": ["States.TaskFailed"], "IntervalSeconds": 30, "MaxAttempts": 1, "BackoffRate": 2.0}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "EvaluateModel": {"Type": "Task", "Resource": "PLACEHOLDER", "Next": "RegisterModel",
                    "Retry": [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"], "IntervalSeconds": 5, "MaxAttempts": 3, "BackoffRate": 2.0}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "RegisterModel": {"Type": "Task", "Resource": "arn:aws:states:::sagemaker:createModel", "End": True,
                    "Retry": [{"ErrorEquals": ["States.TaskFailed"], "IntervalSeconds": 30, "MaxAttempts": 1, "BackoffRate": 2.0}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "BatchPredict": {"Type": "Task", "Resource": "arn:aws:states:::sagemaker:createTransformJob.sync", "Next": "Explainability",
                    "Retry": [{"ErrorEquals": ["States.TaskFailed"], "IntervalSeconds": 30, "MaxAttempts": 1, "BackoffRate": 2.0}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "Explainability": {"Type": "Task", "Resource": "PLACEHOLDER", "Next": "BedrockExplanations",
                    "Retry": [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"], "IntervalSeconds": 5, "MaxAttempts": 3, "BackoffRate": 2.0}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "BedrockExplanations": {"Type": "Task", "Resource": "PLACEHOLDER", "Next": "GenerateReports",
                    "Retry": [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"], "IntervalSeconds": 5, "MaxAttempts": 2, "BackoffRate": 1.5}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "GenerateReports": {"Type": "Task", "Resource": "PLACEHOLDER", "End": True,
                    "Retry": [{"ErrorEquals": ["States.TaskFailed", "States.Timeout"], "IntervalSeconds": 5, "MaxAttempts": 3, "BackoffRate": 2.0}],
                    "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "PipelineFailed"}]},
                "PipelineFailed": {"Type": "Fail", "Cause": "Pipeline execution failed", "Error": "PipelineError"},
            },
        }

        # Mapear estados para ARNs reais dos Lambdas
        state_to_handler = {
            "Ingestion": "ingest",
            "Extraction": "extract",
            "FeatureEngineering": "feature",
            "StoreFeatures": "store",
            "Explainability": "shap",
            "BedrockExplanations": "bedrock",
            "GenerateReports": "report",
        }

        for state_name, handler_name in state_to_handler.items():
            if state_name in definition["States"]:
                state = definition["States"][state_name]
                if state.get("Type") == "Task":
                    fn = self.lambda_functions[handler_name]
                    state["Resource"] = fn.function_arn

        # Role para Step Functions invocar Lambdas e SageMaker
        sfn_role = iam.Role(
            self,
            "StepFunctionsRole",
            role_name="churn-pipeline-stepfunctions-role",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
            description="Role para Step Functions orquestrar o pipeline",
        )

        # Permissão para invocar todas as Lambda functions
        for fn in self.lambda_functions.values():
            fn.grant_invoke(sfn_role)

        # Permissão para SageMaker (Training + Batch Transform)
        sfn_role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:CreateTransformJob",
                    "sagemaker:DescribeTransformJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:StopTransformJob",
                    "sagemaker:CreateModel",
                    "sagemaker:CreateModelPackage",
                ],
                resources=["*"],
            )
        )

        # Permissão para eventos e logs
        sfn_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogs",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogDelivery",
                    "logs:GetLogDelivery",
                    "logs:UpdateLogDelivery",
                    "logs:DeleteLogDelivery",
                    "logs:ListLogDeliveries",
                    "logs:PutResourcePolicy",
                ],
                resources=["*"],
            )
        )

        # Criar state machine via CfnStateMachine (permite ASL customizado)
        state_machine = sfn.CfnStateMachine(
            self,
            "ChurnPipelineStateMachine",
            state_machine_name="churn-prediction-pipeline",
            definition_string=json.dumps(definition),
            role_arn=sfn_role.role_arn,
            state_machine_type="STANDARD",
        )

        return state_machine

    def _create_weekly_schedule(self) -> events.Rule:
        """Cria EventBridge rule com cron semanal (segunda, 08:00 UTC).

        Inicia a state machine em modo 'predict' automaticamente.
        """
        rule = events.Rule(
            self,
            "WeeklyPipelineSchedule",
            rule_name="churn-pipeline-weekly-schedule",
            description=(
                "Execução semanal do pipeline de churn prediction "
                "(segunda às 08:00 UTC)"
            ),
            schedule=events.Schedule.cron(
                minute="0",
                hour="8",
                week_day="MON",
            ),
        )

        # Target: iniciar a state machine com input padrão (modo predict)
        rule.add_target(
            events_targets.SfnStateMachine(
                sfn.StateMachine.from_state_machine_arn(
                    self,
                    "ImportedStateMachine",
                    self.state_machine.attr_arn,
                ),
                input=events.RuleTargetInput.from_object({
                    "mode": "predict",
                    "source": "scheduled",
                }),
            )
        )

        return rule

    def _create_s3_trigger(self) -> None:
        """Configura S3 event notification para iniciar o pipeline via upload.

        Quando um arquivo é carregado no prefixo input/ do bucket,
        uma Lambda é invocada para iniciar a state machine.
        """
        # Lambda trigger que inicia a state machine a partir do evento S3
        trigger_fn = _lambda.Function(
            self,
            "S3TriggerHandler",
            function_name="churn-pipeline-s3-trigger",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_inline(self._s3_trigger_code()),
            timeout=Duration.seconds(30),
            memory_size=128,
            description=(
                "Trigger: inicia pipeline ao detectar upload em input/"
            ),
            role=self._lambda_role,
            environment={
                "STATE_MACHINE_ARN": self.state_machine.attr_arn,
            },
        )

        # Permissão para iniciar a state machine
        trigger_fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="StartExecution",
                effect=iam.Effect.ALLOW,
                actions=["states:StartExecution"],
                resources=[self.state_machine.attr_arn],
            )
        )

        # Adicionar notificação S3 para prefixo input/
        self._bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(trigger_fn),
            s3.NotificationKeyFilter(prefix="input/"),
        )

    @staticmethod
    def _s3_trigger_code() -> str:
        """Código inline da Lambda de trigger S3 → Step Functions."""
        return """
import json
import os
import uuid

import boto3

sfn_client = boto3.client("stepfunctions")
STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]


def handler(event, context):
    \"\"\"Inicia pipeline quando arquivo é enviado ao input/.\"\"\"
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        execution_id = uuid.uuid4().hex[:12]
        input_data = {
            "mode": "predict",
            "source": "s3_upload",
            "trigger_bucket": bucket,
            "trigger_key": key,
            "execution_id": execution_id,
        }

        sfn_client.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=f"s3-trigger-{execution_id}",
            input=json.dumps(input_data),
        )

        print(f"Pipeline iniciado: execution={execution_id}, key={key}")

    return {"statusCode": 200, "body": "OK"}
"""

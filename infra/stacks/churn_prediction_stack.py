"""
Stack CDK completa para a plataforma de predição de churn - Sky Brazil.

Recursos em stack única (evita ciclos de dependência):
- S3 Bucket (dados, modelos, relatórios)
- DynamoDB tables (feature_store, predictions, executions)
- Secrets Manager (NPAW API key)
- IAM Roles (Lambda, SageMaker, ECS)
- Lambda functions (8 handlers do pipeline)
- Step Functions state machine
- EventBridge rule (agendamento semanal)
- S3 event trigger (upload → pipeline)

Sem VPC/NAT Gateway — custo estimado: ~$5-15/mês.
"""

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_dynamodb as dynamodb,
    aws_ecr_assets as ecr_assets,
    aws_events as events,
    aws_events_targets as events_targets,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_secretsmanager as secretsmanager,
    aws_stepfunctions as sfn,
)
from constructs import Construct


class ChurnPredictionStack(Stack):
    """Stack completa da plataforma de predição de churn."""

    HANDLER_NAMES = [
        "ingest", "extract", "feature", "store",
        "predict", "shap", "bedrock", "report",
    ]

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # === Storage ===
        self.bucket = self._create_s3_bucket()
        self.feature_store_table = self._create_dynamo_table(
            "ChurnFeatureStore", "churn_feature_store",
            pk="user_id", sk="version", sk_type=dynamodb.AttributeType.NUMBER,
        )
        self.predictions_table = self._create_dynamo_table(
            "ChurnPredictions", "churn_predictions",
            pk="execution_id", sk="user_id",
        )
        self.executions_table = self._create_dynamo_table(
            "ChurnExecutions", "churn_executions", pk="execution_id",
        )
        self.npaw_secret = secretsmanager.Secret(
            self, "NpawApiKey",
            secret_name="churn-prediction/npaw-api-key",
            description="NPAW API Key - Sky Brazil",
            removal_policy=RemovalPolicy.DESTROY,
        )

        # === IAM Roles ===
        self.lambda_role = self._create_lambda_role()
        self.sagemaker_role = self._create_sagemaker_role()
        self.ecs_task_role = self._create_ecs_role()

        # === Lambda Functions ===
        self.lambda_functions = self._create_lambda_functions()

        # === Step Functions ===
        self.state_machine = self._create_state_machine()

        # === EventBridge (cron semanal) ===
        self._create_weekly_schedule()

        # === S3 Trigger (upload → pipeline) ===
        self._create_s3_trigger()

        # === CloudWatch Log Groups ===
        self._create_log_groups()

    # ------------------------------------------------------------------
    # S3
    # ------------------------------------------------------------------
    def _create_s3_bucket(self) -> s3.Bucket:
        return s3.Bucket(
            self, "Bucket",
            bucket_name="sky-brazil-churn-prediction",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY,
            lifecycle_rules=[
                s3.LifecycleRule(id="raw", prefix="raw_data/",
                    transitions=[s3.Transition(storage_class=s3.StorageClass.INFREQUENT_ACCESS, transition_after=Duration.days(90))],
                    expiration=Duration.days(365)),
                s3.LifecycleRule(id="predictions", prefix="predictions/",
                    transitions=[s3.Transition(storage_class=s3.StorageClass.INFREQUENT_ACCESS, transition_after=Duration.days(90))],
                    expiration=Duration.days(365)),
                s3.LifecycleRule(id="reports", prefix="reports/",
                    transitions=[s3.Transition(storage_class=s3.StorageClass.INFREQUENT_ACCESS, transition_after=Duration.days(180))],
                    expiration=Duration.days(730)),
            ],
        )

    # ------------------------------------------------------------------
    # DynamoDB
    # ------------------------------------------------------------------
    def _create_dynamo_table(
        self, id: str, name: str, pk: str, sk=None,
        sk_type=dynamodb.AttributeType.STRING,
    ) -> dynamodb.Table:
        kwargs = dict(
            table_name=name,
            partition_key=dynamodb.Attribute(name=pk, type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
        )
        if sk:
            kwargs["sort_key"] = dynamodb.Attribute(name=sk, type=sk_type)
        return dynamodb.Table(self, id, **kwargs)

    # ------------------------------------------------------------------
    # IAM
    # ------------------------------------------------------------------
    def _create_lambda_role(self) -> iam.Role:
        role = iam.Role(self, "LambdaRole",
            role_name="churn-prediction-lambda-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name(
            "service-role/AWSLambdaBasicExecutionRole"
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
            resources=[self.bucket.bucket_arn, f"{self.bucket.bucket_arn}/*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query",
                     "dynamodb:BatchWriteItem", "dynamodb:BatchGetItem", "dynamodb:Scan"],
            resources=[self.feature_store_table.table_arn, self.predictions_table.table_arn,
                       self.executions_table.table_arn],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["secretsmanager:GetSecretValue"],
            resources=[self.npaw_secret.secret_arn],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:CreateTransformJob", "sagemaker:DescribeTransformJob",
                     "sagemaker:ListModelPackages", "sagemaker:DescribeModelPackage",
                     "sagemaker:CreateModel"],
            resources=["*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-haiku-*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["cloudwatch:PutMetricData"],
            resources=["*"],
        ))
        return role

    def _create_sagemaker_role(self) -> iam.Role:
        role = iam.Role(self, "SageMakerRole",
            role_name="churn-prediction-sagemaker-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )
        role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            resources=[self.bucket.bucket_arn, f"{self.bucket.bucket_arn}/*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["ecr:GetAuthorizationToken", "ecr:BatchGetImage",
                     "ecr:GetDownloadUrlForLayer"],
            resources=["*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:CreateModelPackage", "sagemaker:ListModelPackages",
                     "sagemaker:DescribeModelPackage", "sagemaker:UpdateModelPackage"],
            resources=["*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            resources=[f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*"],
        ))
        return role

    def _create_ecs_role(self) -> iam.Role:
        role = iam.Role(self, "EcsRole",
            role_name="churn-prediction-ecs-task-role",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:ListBucket"],
            resources=[self.bucket.bucket_arn, f"{self.bucket.bucket_arn}/*"],
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan"],
            resources=[self.predictions_table.table_arn, self.executions_table.table_arn,
                       self.feature_store_table.table_arn],
        ))
        return role

    # ------------------------------------------------------------------
    # Lambda Functions
    # ------------------------------------------------------------------
    def _create_lambda_functions(self) -> dict[str, _lambda.Function]:
        # Layer com dependências (pydantic, pyyaml, aiohttp, numpy, pandas)
        deps_layer = _lambda.LayerVersion(self, "DepsLayer",
            layer_version_name="churn-prediction-deps",
            code=_lambda.Code.from_asset("layers/dependencies"),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_11],
            description="pydantic, pyyaml, aiohttp, numpy, pandas para Lambda handlers",
        )

        functions: dict[str, _lambda.Function] = {}
        common_env = {
            "BUCKET_NAME": self.bucket.bucket_name,
            "FEATURE_STORE_TABLE": self.feature_store_table.table_name,
            "PREDICTIONS_TABLE": self.predictions_table.table_name,
            "EXECUTIONS_TABLE": self.executions_table.table_name,
        }
        handler_config = {
            "ingest":  {"timeout": 60,  "memory": 256},
            "extract": {"timeout": 900, "memory": 512},
            "feature": {"timeout": 600, "memory": 1024},
            "store":   {"timeout": 300, "memory": 256},
            "predict": {"timeout": 900, "memory": 512},
            "shap":    {"timeout": 600, "memory": 2048},
            "bedrock": {"timeout": 600, "memory": 512},
            "report":  {"timeout": 300, "memory": 512},
        }

        for name in self.HANDLER_NAMES:
            cfg = handler_config[name]

            # SHAP usa Lambda Docker (lib SHAP é pesada demais para Layer)
            # Imagem é buildada via CodeBuild (scripts/build-shap-image.ps1)
            # e armazenada no ECR. CDK referencia a imagem existente.
            if name == "shap":
                ecr_repo = f"{self.account}.dkr.ecr.{self.region}.amazonaws.com/churn-pipeline-shap:latest"
                fn = _lambda.DockerImageFunction(self, f"Fn-{name}",
                    function_name=f"churn-pipeline-{name}",
                    code=_lambda.DockerImageCode.from_ecr(
                        repository=__import__("aws_cdk").aws_ecr.Repository.from_repository_name(
                            self, "ShapEcrRepo", "churn-pipeline-shap"
                        ),
                        tag_or_digest="latest",
                    ),
                    timeout=Duration.seconds(cfg["timeout"]),
                    memory_size=cfg["memory"],
                    role=self.lambda_role,
                    environment=common_env,
                )
            else:
                fn = _lambda.Function(self, f"Fn-{name}",
                    function_name=f"churn-pipeline-{name}",
                    runtime=_lambda.Runtime.PYTHON_3_11,
                    handler=f"src.orchestrator.handlers.{name}_handler.handler",
                    code=_lambda.Code.from_asset("..", exclude=[
                        "infra/*", "docs/*", "tests/*", ".git/*",
                        ".hypothesis/*", "__pycache__/*", ".venv/*", "cdk.out/*",
                    ]),
                    timeout=Duration.seconds(cfg["timeout"]),
                    memory_size=cfg["memory"],
                    role=self.lambda_role,
                    layers=[deps_layer],
                    environment=common_env,
                )

            functions[name] = fn
        return functions

    # ------------------------------------------------------------------
    # Step Functions
    # ------------------------------------------------------------------
    def _create_state_machine(self) -> sfn.CfnStateMachine:
        import json, copy, sys
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent))
        from src.orchestrator.step_functions import PIPELINE_DEFINITION

        definition = copy.deepcopy(PIPELINE_DEFINITION)

        # Mapear estados para Lambda ARNs
        state_handler_map = {
            "Ingestion": "ingest", "Extraction": "extract",
            "FeatureEngineering": "feature", "StoreFeatures": "store",
            "Training": "predict", "EvaluateModel": "predict",
            "RegisterModel": "predict", "BatchPredict": "predict",
            "Explainability": "shap", "BedrockExplanations": "bedrock",
            "GenerateReports": "report",
        }
        for state_name, handler_name in state_handler_map.items():
            if state_name in definition["States"]:
                state = definition["States"][state_name]
                if state.get("Type") == "Task":
                    state["Resource"] = self.lambda_functions[handler_name].function_arn

        # Role para Step Functions
        sfn_role = iam.Role(self, "SfnRole",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
        )
        for fn in self.lambda_functions.values():
            fn.grant_invoke(sfn_role)
        sfn_role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob",
                     "sagemaker:CreateTransformJob", "sagemaker:DescribeTransformJob",
                     "sagemaker:CreateModel"],
            resources=["*"],
        ))

        return sfn.CfnStateMachine(self, "Pipeline",
            state_machine_name="churn-prediction-pipeline",
            definition_string=json.dumps(definition),
            role_arn=sfn_role.role_arn,
            state_machine_type="STANDARD",
        )

    # ------------------------------------------------------------------
    # EventBridge (cron semanal - segunda 08:00 UTC)
    # ------------------------------------------------------------------
    def _create_weekly_schedule(self) -> None:
        rule = events.Rule(self, "WeeklySchedule",
            rule_name="churn-pipeline-weekly",
            schedule=events.Schedule.cron(minute="0", hour="8", week_day="MON"),
        )
        rule.add_target(events_targets.SfnStateMachine(
            sfn.StateMachine.from_state_machine_arn(
                self, "ImportedSM", self.state_machine.attr_arn,
            ),
            input=events.RuleTargetInput.from_object({"mode": "predict", "source": "scheduled"}),
        ))

    # ------------------------------------------------------------------
    # S3 Trigger (upload em input/ → inicia pipeline)
    # ------------------------------------------------------------------
    def _create_s3_trigger(self) -> None:
        # Role separado para o trigger (evita circular dependency com bucket)
        trigger_role = iam.Role(self, "S3TriggerRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        trigger_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name(
            "service-role/AWSLambdaBasicExecutionRole"
        ))
        trigger_role.add_to_policy(iam.PolicyStatement(
            actions=["states:StartExecution"],
            resources=[self.state_machine.attr_arn],
        ))

        trigger_fn = _lambda.Function(self, "S3Trigger",
            function_name="churn-pipeline-s3-trigger",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_inline(
                'import json, os, uuid, boto3\n'
                'sfn = boto3.client("stepfunctions")\n'
                'SM_ARN = os.environ["STATE_MACHINE_ARN"]\n'
                'def handler(event, ctx):\n'
                '    for r in event.get("Records", []):\n'
                '        key = r["s3"]["object"]["key"]\n'
                '        eid = uuid.uuid4().hex[:12]\n'
                '        sfn.start_execution(stateMachineArn=SM_ARN, name=f"s3-{eid}",\n'
                '            input=json.dumps({"mode":"predict","source":"s3","trigger_key":key}))\n'
                '    return {"statusCode": 200}\n'
            ),
            timeout=Duration.seconds(30),
            environment={"STATE_MACHINE_ARN": self.state_machine.attr_arn},
            role=trigger_role,
        )
        self.bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(trigger_fn),
            s3.NotificationKeyFilter(prefix="input/"),
        )

    # ------------------------------------------------------------------
    # CloudWatch Log Groups
    # ------------------------------------------------------------------
    def _create_log_groups(self) -> None:
        stages = ["extraction", "feature-engineering", "ml-inference",
                  "explainability", "bedrock-explanation", "report-generation", "dashboard"]
        for stage in stages:
            logs.LogGroup(self, f"Log-{stage}",
                log_group_name=f"/churn-prediction/{stage}",
                retention=logs.RetentionDays.THREE_MONTHS,
                removal_policy=RemovalPolicy.DESTROY,
            )

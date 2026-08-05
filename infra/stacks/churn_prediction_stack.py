"""
Stack CDK principal para a plataforma de predição de churn.

Recursos provisionados:
- S3 Bucket com estrutura de prefixos para dados, modelos e relatórios
- DynamoDB tables: churn_feature_store, churn_predictions, churn_executions
- Secrets Manager para NPAW API key
- IAM Roles (least privilege) para Lambda, SageMaker, ECS
- VPC endpoints para serviços AWS
"""

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class ChurnPredictionStack(Stack):
    """Stack principal com recursos base da plataforma de predição de churn."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ============================================================
        # S3 Bucket - Armazenamento de dados, modelos e relatórios
        # ============================================================
        self.bucket = self._create_s3_bucket()

        # ============================================================
        # DynamoDB Tables
        # ============================================================
        self.feature_store_table = self._create_feature_store_table()
        self.predictions_table = self._create_predictions_table()
        self.executions_table = self._create_executions_table()

        # ============================================================
        # Secrets Manager - NPAW API Key
        # ============================================================
        self.npaw_secret = self._create_npaw_secret()

        # ============================================================
        # VPC com endpoints para serviços AWS
        # ============================================================
        self.vpc = self._create_vpc_with_endpoints()

        # ============================================================
        # IAM Roles (least privilege)
        # ============================================================
        self.lambda_execution_role = self._create_lambda_execution_role()
        self.sagemaker_execution_role = self._create_sagemaker_execution_role()
        self.ecs_task_role = self._create_ecs_task_role()

    # ------------------------------------------------------------------
    # S3 Bucket
    # ------------------------------------------------------------------
    def _create_s3_bucket(self) -> s3.Bucket:
        """Cria bucket S3 com lifecycle rules e encryption."""
        bucket = s3.Bucket(
            self,
            "ChurnPredictionBucket",
            bucket_name="sky-brazil-churn-prediction",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY,
            lifecycle_rules=[
                # Dados brutos: mover para IA após 90 dias, expirar após 365
                s3.LifecycleRule(
                    id="raw-data-lifecycle",
                    prefix="raw_data/",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(90),
                        ),
                    ],
                    expiration=Duration.days(365),
                ),
                # Features: mover para IA após 180 dias
                s3.LifecycleRule(
                    id="features-lifecycle",
                    prefix="features/",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(180),
                        ),
                    ],
                ),
                # Modelos: reter indefinidamente (sem expiração)
                s3.LifecycleRule(
                    id="models-lifecycle",
                    prefix="models/",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(365),
                        ),
                    ],
                ),
                # Relatórios: expirar após 2 anos
                s3.LifecycleRule(
                    id="reports-lifecycle",
                    prefix="reports/",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(180),
                        ),
                    ],
                    expiration=Duration.days(730),
                ),
                # Predictions: mover para IA após 90 dias
                s3.LifecycleRule(
                    id="predictions-lifecycle",
                    prefix="predictions/",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(90),
                        ),
                    ],
                    expiration=Duration.days(365),
                ),
            ],
        )

        return bucket

    # ------------------------------------------------------------------
    # DynamoDB Tables
    # ------------------------------------------------------------------
    def _create_feature_store_table(self) -> dynamodb.Table:
        """
        Tabela churn_feature_store.
        PK: user_id (String), SK: version (Number)
        Armazena Feature Vectors versionados e imutáveis.
        """
        table = dynamodb.Table(
            self,
            "ChurnFeatureStoreTable",
            table_name="churn_feature_store",
            partition_key=dynamodb.Attribute(
                name="user_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="version",
                type=dynamodb.AttributeType.NUMBER,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True,
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        return table

    def _create_predictions_table(self) -> dynamodb.Table:
        """
        Tabela churn_predictions.
        PK: execution_id (String), SK: user_id (String)
        Armazena resultados de predição por execução.
        """
        table = dynamodb.Table(
            self,
            "ChurnPredictionsTable",
            table_name="churn_predictions",
            partition_key=dynamodb.Attribute(
                name="execution_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="user_id",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True,
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        return table

    def _create_executions_table(self) -> dynamodb.Table:
        """
        Tabela churn_executions.
        PK: execution_id (String)
        Armazena metadados das execuções do pipeline.
        """
        table = dynamodb.Table(
            self,
            "ChurnExecutionsTable",
            table_name="churn_executions",
            partition_key=dynamodb.Attribute(
                name="execution_id",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True,
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        return table

    # ------------------------------------------------------------------
    # Secrets Manager
    # ------------------------------------------------------------------
    def _create_npaw_secret(self) -> secretsmanager.Secret:
        """Cria secret para a API Key da NPAW."""
        secret = secretsmanager.Secret(
            self,
            "NpawApiKeySecret",
            secret_name="churn-prediction/npaw-api-key",
            description="API Key para autenticação na NPAW (Sky Brazil)",
            removal_policy=RemovalPolicy.DESTROY,
        )

        return secret

    # ------------------------------------------------------------------
    # VPC com Endpoints
    # ------------------------------------------------------------------
    def _create_vpc_with_endpoints(self) -> ec2.Vpc:
        """Cria VPC com endpoints para serviços AWS (reduz custos e melhora segurança)."""
        vpc = ec2.Vpc(
            self,
            "ChurnPredictionVpc",
            vpc_name="churn-prediction-vpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
                ),
            ],
        )

        # VPC Endpoints para serviços AWS (Gateway endpoints - sem custo adicional)
        vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )

        vpc.add_gateway_endpoint(
            "DynamoDBEndpoint",
            service=ec2.GatewayVpcEndpointAwsService.DYNAMODB,
        )

        # VPC Endpoints (Interface) para serviços AWS
        vpc.add_interface_endpoint(
            "SecretsManagerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
            private_dns_enabled=True,
        )

        vpc.add_interface_endpoint(
            "SageMakerRuntimeEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_RUNTIME,
            private_dns_enabled=True,
        )

        vpc.add_interface_endpoint(
            "SageMakerApiEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_API,
            private_dns_enabled=True,
        )

        vpc.add_interface_endpoint(
            "BedrockRuntimeEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
            private_dns_enabled=True,
        )

        vpc.add_interface_endpoint(
            "CloudWatchLogsEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
            private_dns_enabled=True,
        )

        vpc.add_interface_endpoint(
            "StepFunctionsEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.STEP_FUNCTIONS,
            private_dns_enabled=True,
        )

        return vpc

    # ------------------------------------------------------------------
    # IAM Roles (Least Privilege)
    # ------------------------------------------------------------------
    def _create_lambda_execution_role(self) -> iam.Role:
        """
        IAM Role para funções Lambda do pipeline.
        Acesso mínimo necessário: DynamoDB, S3, Secrets Manager, CloudWatch Logs,
        SageMaker (invoke), Bedrock (invoke), Step Functions.
        """
        role = iam.Role(
            self,
            "LambdaExecutionRole",
            role_name="churn-prediction-lambda-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Role para Lambda functions do pipeline de churn prediction",
        )

        # CloudWatch Logs - criar e escrever logs
        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogs",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/churn-prediction/*",
                ],
            )
        )

        # S3 - leitura e escrita no bucket do projeto
        role.add_to_policy(
            iam.PolicyStatement(
                sid="S3Access",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:DeleteObject",
                ],
                resources=[
                    self.bucket.bucket_arn,
                    f"{self.bucket.bucket_arn}/*",
                ],
            )
        )

        # DynamoDB - leitura e escrita nas tabelas do projeto
        role.add_to_policy(
            iam.PolicyStatement(
                sid="DynamoDBAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:Query",
                    "dynamodb:BatchWriteItem",
                    "dynamodb:BatchGetItem",
                ],
                resources=[
                    self.feature_store_table.table_arn,
                    self.predictions_table.table_arn,
                    self.executions_table.table_arn,
                ],
            )
        )

        # Secrets Manager - ler a chave NPAW
        role.add_to_policy(
            iam.PolicyStatement(
                sid="SecretsManagerRead",
                effect=iam.Effect.ALLOW,
                actions=[
                    "secretsmanager:GetSecretValue",
                ],
                resources=[
                    self.npaw_secret.secret_arn,
                ],
            )
        )

        # SageMaker - invocar Batch Transform e consultar Model Registry
        role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerInvoke",
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:CreateTransformJob",
                    "sagemaker:DescribeTransformJob",
                    "sagemaker:ListModelPackages",
                    "sagemaker:DescribeModelPackage",
                ],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "aws:RequestedRegion": self.region,
                    }
                },
            )
        )

        # Bedrock - invocar modelo para explicações
        role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockInvoke",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                ],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-haiku-*",
                ],
            )
        )

        # CloudWatch Metrics - publicar métricas customizadas
        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchMetrics",
                effect=iam.Effect.ALLOW,
                actions=[
                    "cloudwatch:PutMetricData",
                ],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "cloudwatch:namespace": "ChurnPrediction",
                    }
                },
            )
        )

        # VPC - acesso à rede para Lambda dentro da VPC
        role.add_to_policy(
            iam.PolicyStatement(
                sid="VpcNetworkInterface",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ec2:CreateNetworkInterface",
                    "ec2:DescribeNetworkInterfaces",
                    "ec2:DeleteNetworkInterface",
                ],
                resources=["*"],
            )
        )

        return role

    def _create_sagemaker_execution_role(self) -> iam.Role:
        """
        IAM Role para SageMaker Training Jobs e Batch Transform.
        Acesso: S3 (dados e modelos), CloudWatch Logs, ECR (containers).
        """
        role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name="churn-prediction-sagemaker-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Role para SageMaker Training e Batch Transform",
        )

        # S3 - leitura de dados de treino e escrita de artefatos de modelo
        role.add_to_policy(
            iam.PolicyStatement(
                sid="S3Access",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                ],
                resources=[
                    self.bucket.bucket_arn,
                    f"{self.bucket.bucket_arn}/*",
                ],
            )
        )

        # CloudWatch Logs - logs de training jobs
        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogs",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*",
                ],
            )
        )

        # ECR - pull de containers (XGBoost, LightGBM built-in)
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ECRAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchCheckLayerAvailability",
                ],
                resources=["*"],
            )
        )

        # SageMaker Model Registry - registrar e consultar modelos
        role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerModelRegistry",
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:CreateModelPackage",
                    "sagemaker:CreateModelPackageGroup",
                    "sagemaker:DescribeModelPackage",
                    "sagemaker:DescribeModelPackageGroup",
                    "sagemaker:ListModelPackages",
                    "sagemaker:UpdateModelPackage",
                ],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:model-package-group/churn-prediction-models",
                    f"arn:aws:sagemaker:{self.region}:{self.account}:model-package/churn-prediction-models/*",
                ],
            )
        )

        # CloudWatch Metrics - publicar métricas de treinamento
        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchMetrics",
                effect=iam.Effect.ALLOW,
                actions=[
                    "cloudwatch:PutMetricData",
                ],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "cloudwatch:namespace": "ChurnPrediction",
                    }
                },
            )
        )

        # VPC - caso SageMaker execute dentro da VPC
        role.add_to_policy(
            iam.PolicyStatement(
                sid="VpcAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "ec2:CreateNetworkInterface",
                    "ec2:CreateNetworkInterfacePermission",
                    "ec2:DescribeNetworkInterfaces",
                    "ec2:DeleteNetworkInterface",
                    "ec2:DescribeVpcs",
                    "ec2:DescribeSubnets",
                    "ec2:DescribeSecurityGroups",
                ],
                resources=["*"],
            )
        )

        return role

    def _create_ecs_task_role(self) -> iam.Role:
        """
        IAM Role para ECS Fargate Task (Dashboard Streamlit).
        Acesso: S3 (leitura relatórios), DynamoDB (leitura predições), CloudWatch Logs.
        """
        role = iam.Role(
            self,
            "EcsTaskRole",
            role_name="churn-prediction-ecs-task-role",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Role para ECS Fargate task do dashboard Streamlit",
        )

        # S3 - leitura de relatórios e predições
        role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ReadAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:ListBucket",
                ],
                resources=[
                    self.bucket.bucket_arn,
                    f"{self.bucket.bucket_arn}/reports/*",
                    f"{self.bucket.bucket_arn}/predictions/*",
                ],
            )
        )

        # DynamoDB - leitura das tabelas de predição e execuções
        role.add_to_policy(
            iam.PolicyStatement(
                sid="DynamoDBReadAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:Query",
                    "dynamodb:Scan",
                    "dynamodb:BatchGetItem",
                ],
                resources=[
                    self.predictions_table.table_arn,
                    self.executions_table.table_arn,
                    self.feature_store_table.table_arn,
                ],
            )
        )

        # CloudWatch Logs - logs do dashboard
        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogs",
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/churn-prediction/dashboard*",
                ],
            )
        )

        return role

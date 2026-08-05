"""
Stack CDK para o Dashboard ECS Fargate (Streamlit).

Recursos provisionados:
- ECS Fargate Service com container Streamlit (porta 8501)
- Application Load Balancer com autenticação Cognito
- CloudWatch Log Groups para cada estágio do pipeline
- CloudWatch Alarms para métricas customizadas (InferenceTime,
  PredictionFailures, FeatureDriftDetected)
"""

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_cloudwatch as cloudwatch,
    aws_cognito as cognito,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elbv2,
    aws_elasticloadbalancingv2_actions as elb_actions,
    aws_iam as iam,
    aws_logs as logs,
)
from constructs import Construct


class DashboardStack(Stack):
    """Stack do Dashboard Streamlit com ECS Fargate, ALB e Cognito."""

    # Estágios do pipeline para criação de Log Groups
    PIPELINE_STAGES = [
        "extraction",
        "feature-engineering",
        "ml-inference",
        "explainability",
        "bedrock-explanation",
        "report-generation",
        "dashboard",
    ]

    # Namespace de métricas customizadas
    METRICS_NAMESPACE = "ChurnPrediction"

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        ecs_task_role: iam.IRole,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ============================================================
        # CloudWatch Log Groups - Um por estágio do pipeline
        # ============================================================
        self.log_groups = self._create_log_groups()

        # ============================================================
        # Cognito User Pool - Autenticação do Dashboard
        # ============================================================
        self.user_pool = self._create_cognito_user_pool()

        # ============================================================
        # ECS Cluster + Fargate Service (Streamlit) — VPC pública mínima
        # ============================================================
        vpc = ec2.Vpc(
            self,
            "DashboardVpc",
            vpc_name="churn-dashboard-vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
            ],
        )

        self.cluster = ecs.Cluster(
            self,
            "DashboardCluster",
            cluster_name="churn-dashboard-cluster",
            vpc=vpc,
        )

        self.fargate_service = self._create_fargate_service(
            vpc=vpc,
            ecs_task_role=ecs_task_role,
        )

        # ============================================================
        # ALB + Cognito Authentication
        # ============================================================
        self._configure_cognito_auth_on_alb()

        # ============================================================
        # CloudWatch Alarms - Métricas customizadas
        # ============================================================
        self.alarms = self._create_cloudwatch_alarms()

    # ------------------------------------------------------------------
    # CloudWatch Log Groups
    # ------------------------------------------------------------------
    def _create_log_groups(self) -> dict[str, logs.LogGroup]:
        """Cria Log Groups para cada estágio do pipeline."""
        log_groups: dict[str, logs.LogGroup] = {}

        for stage in self.PIPELINE_STAGES:
            log_group = logs.LogGroup(
                self,
                f"LogGroup-{stage}",
                log_group_name=f"/churn-prediction/{stage}",
                retention=logs.RetentionDays.THREE_MONTHS,
                removal_policy=RemovalPolicy.DESTROY,
            )
            log_groups[stage] = log_group

        return log_groups

    # ------------------------------------------------------------------
    # Cognito User Pool
    # ------------------------------------------------------------------
    def _create_cognito_user_pool(self) -> cognito.UserPool:
        """Cria User Pool Cognito para autenticação do dashboard."""
        user_pool = cognito.UserPool(
            self,
            "DashboardUserPool",
            user_pool_name="churn-dashboard-users",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=False,
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Domínio Cognito para hosted UI (login)
        user_pool.add_domain(
            "DashboardCognitoDomain",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix="churn-dashboard-sky",
            ),
        )

        return user_pool

    # ------------------------------------------------------------------
    # ECS Fargate Service (Streamlit)
    # ------------------------------------------------------------------
    def _create_fargate_service(
        self,
        vpc: ec2.IVpc,
        ecs_task_role: iam.IRole,
    ) -> ecs_patterns.ApplicationLoadBalancedFargateService:
        """Cria Fargate Service com ALB para o Streamlit dashboard."""
        task_image_opts = (
            ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_registry(
                    "amazon/amazon-ecs-sample"
                ),
                container_port=8501,
                task_role=ecs_task_role,
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="dashboard",
                    log_group=self.log_groups["dashboard"],
                ),
                environment={
                    "STREAMLIT_SERVER_PORT": "8501",
                    "STREAMLIT_SERVER_ADDRESS": "0.0.0.0",
                    "STREAMLIT_SERVER_HEADLESS": "true",
                    "AWS_DEFAULT_REGION": self.region,
                },
            )
        )

        fargate_service = (
            ecs_patterns.ApplicationLoadBalancedFargateService(
                self,
                "DashboardFargateService",
                cluster=self.cluster,
                cpu=512,
                memory_limit_mib=1024,
                desired_count=1,
                task_image_options=task_image_opts,
                public_load_balancer=True,
                assign_public_ip=True,
                listener_port=443,
            )
        )

        # Health check para o Streamlit
        fargate_service.target_group.configure_health_check(
            path="/_stcore/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(10),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        # Auto Scaling
        scaling = fargate_service.service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=3,
        )
        scaling.scale_on_cpu_utilization(
            "CpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(300),
            scale_out_cooldown=Duration.seconds(60),
        )

        return fargate_service

    # ------------------------------------------------------------------
    # Cognito Auth no ALB
    # ------------------------------------------------------------------
    def _configure_cognito_auth_on_alb(self) -> None:
        """Configura autenticação Cognito no ALB listener."""
        # Client para o ALB usar na autenticação
        self.user_pool_client = self.user_pool.add_client(
            "DashboardALBClient",
            user_pool_client_name="churn-dashboard-alb-client",
            generate_secret=True,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(
                    authorization_code_grant=True,
                ),
                scopes=[cognito.OAuthScope.OPENID],
                callback_urls=[
                    "https://placeholder.example.com/oauth2/idpresponse",
                ],
            ),
            supported_identity_providers=[
                cognito.UserPoolClientIdentityProvider.COGNITO,
            ],
        )

        # Adicionar regra de autenticação Cognito ao listener
        listener = self.fargate_service.listener

        # Remover a regra default e adicionar com Cognito auth
        listener.add_action(
            "CognitoAuth",
            priority=1,
            conditions=[
                elbv2.ListenerCondition.path_patterns(["/*"]),
            ],
            action=elb_actions.AuthenticateCognitoAction(
                user_pool=self.user_pool,
                user_pool_client=self.user_pool_client,
                user_pool_domain=self.user_pool.node.find_child(
                    "DashboardCognitoDomain"
                ),
                next=elbv2.ListenerAction.forward(
                    target_groups=[
                        self.fargate_service.target_group
                    ]
                ),
            ),
        )

    # ------------------------------------------------------------------
    # CloudWatch Alarms
    # ------------------------------------------------------------------
    def _create_cloudwatch_alarms(self) -> dict[str, cloudwatch.Alarm]:
        """Cria alarmes para métricas customizadas do pipeline."""
        alarms: dict[str, cloudwatch.Alarm] = {}

        # Alarm: InferenceTime > 5000ms
        alarms["inference_time"] = cloudwatch.Alarm(
            self,
            "InferenceTimeAlarm",
            alarm_name="ChurnPrediction-InferenceTime-High",
            alarm_description=(
                "Tempo de inferência excedeu 5000ms"
            ),
            metric=cloudwatch.Metric(
                namespace=self.METRICS_NAMESPACE,
                metric_name="InferenceTime",
                statistic="Average",
                period=Duration.minutes(5),
            ),
            threshold=5000,
            evaluation_periods=2,
            comparison_operator=(
                cloudwatch.ComparisonOperator
                .GREATER_THAN_THRESHOLD
            ),
            treat_missing_data=(
                cloudwatch.TreatMissingData.NOT_BREACHING
            ),
        )

        # Alarm: PredictionFailures > 5% do batch
        alarms["prediction_failures"] = cloudwatch.Alarm(
            self,
            "PredictionFailuresAlarm",
            alarm_name="ChurnPrediction-PredictionFailures-High",
            alarm_description=(
                "Taxa de falhas de predição excedeu 5% do batch"
            ),
            metric=cloudwatch.MathExpression(
                expression=(
                    "(failures / total) * 100"
                ),
                using_metrics={
                    "failures": cloudwatch.Metric(
                        namespace=self.METRICS_NAMESPACE,
                        metric_name="PredictionFailures",
                        statistic="Sum",
                        period=Duration.minutes(5),
                    ),
                    "total": cloudwatch.Metric(
                        namespace=self.METRICS_NAMESPACE,
                        metric_name="PredictionCount",
                        statistic="Sum",
                        period=Duration.minutes(5),
                    ),
                },
                period=Duration.minutes(5),
            ),
            threshold=5,
            evaluation_periods=1,
            comparison_operator=(
                cloudwatch.ComparisonOperator
                .GREATER_THAN_THRESHOLD
            ),
            treat_missing_data=(
                cloudwatch.TreatMissingData.NOT_BREACHING
            ),
        )

        # Alarm: FeatureDriftDetected > 0
        alarms["feature_drift"] = cloudwatch.Alarm(
            self,
            "FeatureDriftAlarm",
            alarm_name="ChurnPrediction-FeatureDrift-Detected",
            alarm_description=(
                "Drift detectado em features do modelo"
            ),
            metric=cloudwatch.Metric(
                namespace=self.METRICS_NAMESPACE,
                metric_name="FeatureDriftDetected",
                statistic="Sum",
                period=Duration.minutes(5),
            ),
            threshold=0,
            evaluation_periods=1,
            comparison_operator=(
                cloudwatch.ComparisonOperator
                .GREATER_THAN_THRESHOLD
            ),
            treat_missing_data=(
                cloudwatch.TreatMissingData.NOT_BREACHING
            ),
        )

        return alarms

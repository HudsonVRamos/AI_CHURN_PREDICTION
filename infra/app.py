#!/usr/bin/env python3
"""CDK Entry Point - Aplicação principal de infraestrutura.

Sem VPC/NAT Gateway para minimizar custos.
Lambdas fora de VPC (acesso direto a serviços AWS + internet para NPAW API).
ECS Fargate com IP público (sem NAT).
"""

import aws_cdk as cdk

from stacks.churn_prediction_stack import ChurnPredictionStack
from stacks.dashboard_stack import DashboardStack
from stacks.orchestration_stack import OrchestrationStack


app = cdk.App()

env = cdk.Environment(
    account=app.node.try_get_context("account"),
    region=app.node.try_get_context("region") or "us-east-1",
)

# Stack principal com todos os recursos base (sem VPC)
main_stack = ChurnPredictionStack(
    app,
    "ChurnPredictionStack",
    env=env,
    description=(
        "Infraestrutura base para a plataforma de "
        "predição de churn - Sky Brazil"
    ),
)

# Stack do Dashboard (ECS Fargate + ALB + Cognito, sem VPC dedicada)
DashboardStack(
    app,
    "DashboardStack",
    ecs_task_role=main_stack.ecs_task_role,
    env=env,
    description=(
        "Dashboard Streamlit com ECS Fargate, ALB e "
        "Cognito - Sky Brazil Churn Prediction"
    ),
)

# Stack de orquestração (sem VPC nos Lambdas)
OrchestrationStack(
    app,
    "OrchestrationStack",
    env=env,
    bucket=main_stack.bucket,
    lambda_role=main_stack.lambda_execution_role,
    feature_store_table=main_stack.feature_store_table,
    predictions_table=main_stack.predictions_table,
    executions_table=main_stack.executions_table,
    description=(
        "Orquestração do pipeline: Step Functions, Lambdas,"
        " EventBridge, S3 trigger"
    ),
)

app.synth()

#!/usr/bin/env python3
"""CDK Entry Point - Deploy completo em stack única.

Uma única stack para evitar ciclos de dependência entre stacks.
Inclui: S3, DynamoDB, Secrets, IAM, Lambdas, Step Functions, EventBridge.
Sem VPC/NAT para minimizar custos.
"""

import aws_cdk as cdk

from stacks.churn_prediction_stack import ChurnPredictionStack

app = cdk.App()

ChurnPredictionStack(
    app,
    "ChurnPredictionStack",
    env=cdk.Environment(
        account="761018874615",
        region="us-east-1",
    ),
    description="AI Churn Prediction Platform - Sky Brazil (completa)",
)

app.synth()

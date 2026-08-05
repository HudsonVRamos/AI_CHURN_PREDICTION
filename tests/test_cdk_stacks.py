"""Testes de assertion/snapshot para CDK stacks.

Valida que os recursos esperados existem nas stacks com propriedades corretas.

Requirements: 7.1, 9.1, 14.7
"""

from __future__ import annotations

import pytest

# Importação condicional para ambientes sem aws-cdk-lib instalado
cdk = pytest.importorskip("aws_cdk", reason="aws-cdk-lib não disponível")
assertions = pytest.importorskip(
    "aws_cdk.assertions", reason="aws_cdk.assertions não disponível"
)

import aws_cdk as cdk_core
from aws_cdk.assertions import Match, Template

from infra.stacks.churn_prediction_stack import ChurnPredictionStack


@pytest.fixture(scope="module")
def template() -> Template:
    """Sintetiza o ChurnPredictionStack e retorna o Template para assertions."""
    app = cdk_core.App()
    stack = ChurnPredictionStack(
        app,
        "TestChurnPredictionStack",
        env=cdk_core.Environment(account="123456789012", region="us-east-1"),
    )
    return Template.from_stack(stack)


# --------------------------------------------------------------------------
# S3 Bucket (Req 7.1 - configuração via recursos gerenciados)
# --------------------------------------------------------------------------
class TestS3Bucket:
    """Testes para o bucket S3 principal."""

    def test_bucket_exists_with_encryption(self, template: Template):
        """Bucket S3 deve existir com server-side encryption habilitada."""
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "BucketName": "sky-brazil-churn-prediction",
                "BucketEncryption": Match.object_like({
                    "ServerSideEncryptionConfiguration": Match.any_value(),
                }),
            },
        )

    def test_bucket_blocks_public_access(self, template: Template):
        """Bucket deve bloquear acesso público."""
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "PublicAccessBlockConfiguration": {
                    "BlockPublicAcls": True,
                    "BlockPublicPolicy": True,
                    "IgnorePublicAcls": True,
                    "RestrictPublicBuckets": True,
                },
            },
        )

    def test_bucket_has_versioning(self, template: Template):
        """Bucket deve ter versionamento habilitado."""
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "VersioningConfiguration": {"Status": "Enabled"},
            },
        )

    def test_bucket_has_lifecycle_rules(self, template: Template):
        """Bucket deve ter regras de lifecycle configuradas."""
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "LifecycleConfiguration": Match.object_like({
                    "Rules": Match.any_value(),
                }),
            },
        )


# --------------------------------------------------------------------------
# DynamoDB Tables (Req 9.1 - Feature Store com versionamento)
# --------------------------------------------------------------------------
class TestDynamoDBTables:
    """Testes para as 3 tabelas DynamoDB."""

    def test_feature_store_table_exists(self, template: Template):
        """Tabela churn_feature_store deve existir com PK=user_id, SK=version."""
        template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": "churn_feature_store",
                "KeySchema": [
                    {"AttributeName": "user_id", "KeyType": "HASH"},
                    {"AttributeName": "version", "KeyType": "RANGE"},
                ],
                "AttributeDefinitions": Match.array_with([
                    {"AttributeName": "user_id", "AttributeType": "S"},
                    {"AttributeName": "version", "AttributeType": "N"},
                ]),
                "BillingMode": "PAY_PER_REQUEST",
            },
        )

    def test_predictions_table_exists(self, template: Template):
        """Tabela churn_predictions deve existir com PK=execution_id, SK=user_id."""
        template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": "churn_predictions",
                "KeySchema": [
                    {"AttributeName": "execution_id", "KeyType": "HASH"},
                    {"AttributeName": "user_id", "KeyType": "RANGE"},
                ],
                "AttributeDefinitions": Match.array_with([
                    {"AttributeName": "execution_id", "AttributeType": "S"},
                    {"AttributeName": "user_id", "AttributeType": "S"},
                ]),
                "BillingMode": "PAY_PER_REQUEST",
            },
        )

    def test_executions_table_exists(self, template: Template):
        """Tabela churn_executions deve existir com PK=execution_id."""
        template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": "churn_executions",
                "KeySchema": [
                    {"AttributeName": "execution_id", "KeyType": "HASH"},
                ],
                "AttributeDefinitions": Match.array_with([
                    {"AttributeName": "execution_id", "AttributeType": "S"},
                ]),
                "BillingMode": "PAY_PER_REQUEST",
            },
        )

    def test_three_dynamo_tables_total(self, template: Template):
        """Devem existir exatamente 3 tabelas DynamoDB."""
        template.resource_count_is("AWS::DynamoDB::Table", 3)

    def test_feature_store_has_point_in_time_recovery(self, template: Template):
        """Feature store deve ter point-in-time recovery habilitado."""
        template.has_resource_properties(
            "AWS::DynamoDB::Table",
            {
                "TableName": "churn_feature_store",
                "PointInTimeRecoverySpecification": {
                    "PointInTimeRecoveryEnabled": True,
                },
            },
        )


# --------------------------------------------------------------------------
# Secrets Manager (Req 7.1 - credenciais gerenciadas)
# --------------------------------------------------------------------------
class TestSecretsManager:
    """Testes para o secret da API Key NPAW."""

    def test_npaw_secret_exists(self, template: Template):
        """Secret para NPAW API key deve existir com nome correto."""
        template.has_resource_properties(
            "AWS::SecretsManager::Secret",
            {
                "Name": "churn-prediction/npaw-api-key",
                "Description": Match.string_like_regexp(".*NPAW.*"),
            },
        )

    def test_exactly_one_secret(self, template: Template):
        """Deve haver exatamente 1 secret no stack."""
        template.resource_count_is("AWS::SecretsManager::Secret", 1)


# --------------------------------------------------------------------------
# IAM Roles (Req 14.7 - roles para lambda, sagemaker, ecs)
# --------------------------------------------------------------------------
class TestIAMRoles:
    """Testes para os IAM Roles com least privilege."""

    def test_lambda_execution_role_exists(self, template: Template):
        """Role para Lambda deve existir com AssumeRole para lambda.amazonaws.com."""
        template.has_resource_properties(
            "AWS::IAM::Role",
            {
                "RoleName": "churn-prediction-lambda-role",
                "AssumeRolePolicyDocument": Match.object_like({
                    "Statement": Match.array_with([
                        Match.object_like({
                            "Effect": "Allow",
                            "Principal": {"Service": "lambda.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }),
                    ]),
                }),
            },
        )

    def test_sagemaker_execution_role_exists(self, template: Template):
        """Role para SageMaker deve existir com AssumeRole para sagemaker.amazonaws.com."""
        template.has_resource_properties(
            "AWS::IAM::Role",
            {
                "RoleName": "churn-prediction-sagemaker-role",
                "AssumeRolePolicyDocument": Match.object_like({
                    "Statement": Match.array_with([
                        Match.object_like({
                            "Effect": "Allow",
                            "Principal": {"Service": "sagemaker.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }),
                    ]),
                }),
            },
        )

    def test_ecs_task_role_exists(self, template: Template):
        """Role para ECS task deve existir com AssumeRole para ecs-tasks.amazonaws.com."""
        template.has_resource_properties(
            "AWS::IAM::Role",
            {
                "RoleName": "churn-prediction-ecs-task-role",
                "AssumeRolePolicyDocument": Match.object_like({
                    "Statement": Match.array_with([
                        Match.object_like({
                            "Effect": "Allow",
                            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }),
                    ]),
                }),
            },
        )

    def test_at_least_three_roles(self, template: Template):
        """Devem existir pelo menos 3 IAM Roles (lambda, sagemaker, ecs)."""
        resources = template.find_resources("AWS::IAM::Role")
        assert len(resources) >= 3, (
            f"Esperado pelo menos 3 roles, encontrado {len(resources)}"
        )


# VPC removida para reduzir custos — Lambdas sem VPC, ECS na default VPC.

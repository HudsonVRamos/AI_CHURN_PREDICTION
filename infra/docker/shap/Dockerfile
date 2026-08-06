FROM public.ecr.aws/lambda/python:3.11

# Atualizar pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar TUDO de wheels pré-compilados (sem compilar nada do source)
# --only-binary=:all: garante que NENHUM pacote será compilado do source
# Se não houver wheel para a plataforma, vai falhar (melhor que travar compilando)
RUN pip install --no-cache-dir --only-binary=:all: \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    pydantic \
    pyyaml \
    shap

# Copiar código fonte
COPY ./src/ ${LAMBDA_TASK_ROOT}/src/

# Handler
CMD ["src.orchestrator.handlers.shap_handler.handler"]

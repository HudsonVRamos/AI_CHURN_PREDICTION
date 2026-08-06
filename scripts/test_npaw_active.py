"""Teste rápido: buscar users ativos via NPAW API rawdata com filtro amplo."""
import requests
import json
import boto3

client = boto3.client("secretsmanager", region_name="us-east-1")
resp = client.get_secret_value(SecretId="churn-prediction/npaw-api-key")
api_key = resp["SecretString"]

url = "https://api.npaw.com/sky_brazil/rawdata"
headers = {"npaw-api-key": api_key}

# Tentar rawdata com filtro por região (LATAM) para pegar users variados
params = {
    "fromDate": "last7days",
    "limit": "500",
    "offset": "0",
    "orderBy": "end_at",
    "orderDirection": "desc",
}

print(f"Testando: {url}")
print(f"Params: {params}")
r = requests.get(url, headers=headers, params=params, timeout=120)
print(f"Status: {r.status_code}")

if r.status_code == 200:
    data = r.json()
    # Extrair user_ids
    sessions = []
    if isinstance(data, list) and data:
        if "data" in data[0]:
            for d in data[0]["data"]:
                if "values" in d:
                    sessions.extend(d["values"])
    elif isinstance(data, dict) and "data" in data:
        for d in data["data"]:
            if "values" in d:
                sessions.extend(d["values"])

    user_ids = set()
    for s in sessions:
        uid = s.get("user_id", "")
        if uid and isinstance(uid, str) and len(uid) > 10:
            user_ids.add(uid.lower())

    print(f"Sessões retornadas: {len(sessions)}")
    print(f"Users únicos: {len(user_ids)}")
    if user_ids:
        for uid in sorted(user_ids)[:5]:
            print(f"  - {uid}")
else:
    print(f"Erro: {r.text[:500]}")

print(f"Testando: {url}")
print(f"Params: {params}")
r = requests.get(url, headers=headers, params=params, timeout=120)
print(f"Status: {r.status_code}")
print(f"Response ({len(r.text)} chars):")
print(r.text[:3000])

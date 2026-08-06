import requests, boto3

client = boto3.client('secretsmanager', region_name='us-east-1')
resp = client.get_secret_value(SecretId='churn-prediction/npaw-api-key')
api_key = resp['SecretString']

url = 'https://api.npaw.com/sky_brazil/rawdata'
headers = {'npaw-api-key': api_key}
params = {'fromDate': 'last7days', 'limit': '500', 'offset': '0', 'orderBy': 'end_at', 'orderDirection': 'desc'}

r = requests.get(url, headers=headers, params=params, timeout=120, allow_redirects=False)
print(f'Status: {r.status_code}')
if r.status_code in (301, 302, 307, 308):
    loc = r.headers.get("Location", "N/A")
    print(f'Redirect to: {loc}')
else:
    # Verificar history
    r2 = requests.get(url, headers=headers, params=params, timeout=120)
    print(f'Final status: {r2.status_code}')
    print(f'Final URL: {r2.url}')
    if r2.history:
        for h in r2.history:
            print(f'  Hop: {h.status_code} -> {h.headers.get("Location", "?")}')
    # Mostrar resultado
    data = r2.json()
    sessions = []
    if isinstance(data, dict) and "data" in data:
        for d in data["data"]:
            if "values" in d:
                sessions.extend(d["values"])
    user_ids = set()
    for s in sessions:
        uid = s.get("user_id", "")
        if uid and isinstance(uid, str) and len(uid) > 10:
            user_ids.add(uid.lower())
    print(f'Sessions: {len(sessions)}, Users: {len(user_ids)}')

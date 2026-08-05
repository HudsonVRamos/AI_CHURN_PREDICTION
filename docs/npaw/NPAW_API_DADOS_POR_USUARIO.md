# Relatório: Dados Disponíveis por Usuário na API NPAW

> Documento de referência para o projeto AI Churn Prediction.  
> Demonstra quais dados podem ser extraídos da NPAW por user_id individual.  
> Última atualização: 05/08/2026

---

## 1. Como Consultar Dados de um Usuário

### Endpoint
```
GET https://api.npaw.com/{accountCode}/rawdata
```

### Autenticação
```
Header: npaw-api-key: <SUA_API_KEY>
```

### Filtro por User ID
```
filter=[{"name":"uf","rules":{"user_id":["<USER_ID>"]}}]
```

### Exemplo Completo (curl)
```bash
curl -H "npaw-api-key: YOUR_API_KEY" \
  "https://api.npaw.com/sky_brazil/rawdata?fromDate=last6months&filter=%5B%7B%22name%22%3A%22uf%22%2C%22rules%22%3A%7B%22user_id%22%3A%5B%2268adc6fc-7ca1-529b-aaf8-c614e8f1cd32%22%5D%7D%7D%5D&limit=100"
```

### Parâmetros Importantes
| Parâmetro | Tipo | Exemplo | Descrição |
|-----------|------|---------|-----------|
| `accountCode` | path | `sky_brazil` | Código da conta NPAW |
| `fromDate` | query | `last6months`, `last30days`, `2026-01-01` | Data início |
| `toDate` | query | `2026-07-31` | Data fim (exclusivo) |
| `filter` | query | JSON URL-encoded | Filtro por dimensões |
| `limit` | query | `100` | Máximo de registros |
| `offset` | query | `0` | Para paginação |
| `orderBy` | query | `end_at` | Ordenação |
| `orderDirection` | query | `desc` | Direção (asc/desc) |
| `type` | query | `ALL`, `VOD`, `LIVE` | Filtrar tipo conteúdo |

---

## 2. Dados Disponíveis por Sessão (Campos Retornados)

Cada registro na API representa **uma sessão de visualização** do usuário. Abaixo estão todos os campos disponíveis organizados por categoria:

### 2.1 Identificação do Usuário
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `user_id` | string | `68adc6fc-7ca1-529b-aaf8-c614e8f1cd32` | ID único do usuário |
| `user_type` | string | `Registered` | Tipo de usuário |
| `token` | string | `11568340021131342199` | Token da sessão |
| `ip` | string | `2803:9810:5995:7810:...` | Endereço IP |
| `ip_version` | string | `IPv6` | Versão do IP |

### 2.2 Conteúdo Assistido
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `content_channel` | string | `Telefe`, `ESPN Argentina` | Canal assistido |
| `title` | string | `LIVE-DSPORTS (ARG)` | Título principal |
| `title2` | string | `Gran Hermano - Generación Dorada` | Título secundário |
| `content_title_episode` | string | `O'Higgins vs. Boca Juniors` | Episódio/evento |
| `content_tv_show` | string | `Copa Sudamericana` | Programa/show |
| `content_id` | string | `a97d2ee4-2140-467e-...` | ID do conteúdo |
| `content_type` | string | `SPORT`, `EPISODE`, `SHOW`, `LIVE` | Tipo de conteúdo |
| `content_playbacktype` | string | `Live`, `VOD` | Tipo de reprodução |
| `content_genre` | string | `Fútbol`, `Telerrealidad` | Gênero |
| `content_season` | string | `1`, `0` | Temporada |
| `content_saga` | string | | Saga/franquia |
| `content_language` | string | | Idioma do conteúdo |
| `content_subtitles` | string | | Legendas |
| `content_gracenote_id` | string | `SH010001313300000000` | ID Gracenote |
| `content_imdb_id` | string | | ID IMDB |
| `content_package` | string | | Pacote de conteúdo |
| `content_cost` | float | `0.0` | Custo |
| `content_price` | float | `0.0` | Preço |

### 2.3 Métricas de Tempo e Engajamento
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `init_time` | long | `1785458679519` | Timestamp início (Unix ms) |
| `end_at` | string | `2026-07-31 00:44:40` | Data/hora fim formatada |
| `end_at_unix` | long | `1785938967502` | Timestamp fim (Unix ms) |
| `effective_time` | int | `402776` | **Tempo efetivo assistido (ms)** |
| `viewtime` | int | `1064` | Tempo total de view (s) |
| `duration` | int | `3603` | Duração do conteúdo (s) |
| `session_time` | int | `1` | Tempo de sessão |
| `pause_time` | int | `0` | Tempo em pausa (ms) |
| `pause_count` | int | `0` | Quantidade de pausas |
| `seek_time` | int | `0` | Tempo em seek (ms) |
| `seek_count` | int | `0` | Quantidade de seeks |
| `ads_time` | int | `0` | Tempo em anúncios (ms) |
| `ads_effective_time` | int | `0` | Tempo efetivo de ads |

### 2.4 Qualidade de Streaming (QoE)
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `avg_bitrate` | int | `3699968` | Bitrate médio (bps) |
| `avg_bitrate_count` | int | `111` | Amostras de bitrate |
| `resolution` | string | `1280x720` | Resolução do vídeo |
| `avg_fps` | int | `29` | FPS médio |
| `avg_fps_count` | int | `78` | Amostras de FPS |
| `avg_latency` | int | `115508` | Latência média (ms) |
| `avg_latency_count` | int | `110` | Amostras de latência |
| `buffer_ratio` | float | `0.0` | Taxa de buffering |
| `buffer_underruns` | int | `0` | Quantidade de rebuffers |
| `buffer_underrun_total` | int | `0` | Tempo total de buffer (ms) |
| `time_between_buffers` | int | `0` | Tempo entre buffers |
| `dropped_frames` | int | `3` | Frames perdidos |
| `startup_time` | int | `0` | Tempo de startup |
| `time_to_first_frame` | int | `0` | Tempo até primeiro frame |
| `player_startup_time` | int | `0` | Tempo de startup do player |
| `throughput` | int | `0` | Throughput |
| `throughput_count` | int | `0` | Amostras de throughput |
| `cdn_downloaded_traffic` | long | `1107098221` | Tráfego baixado via CDN (bytes) |
| `p2p_downloaded_traffic` | int | `0` | Tráfego P2P (bytes) |

### 2.5 Dispositivo e App
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `device.device_model` | string | `UnionTV`, `SM-A045M` | Modelo do dispositivo |
| `device.device_type` | string | `SmartPhone`, `SmartTV` | Tipo de dispositivo |
| `device.device_vendor` | string | `samsung`, `motorola` | Fabricante |
| `device.os` | string | `Android` | Sistema operacional |
| `device.os_version` | string | `Android 14`, `Android 16` | Versão do SO |
| `device.browser` | string | `Android Browser` | Browser |
| `device.browser_ver` | string | `Android Browser` | Versão do browser |
| `device.user_agent` | string | `okhttp/4.11.0 (...)` | User agent completo |
| `device_name` | string | `Android` | Nome do device |
| `app_name` | string | `DGO Android` | Nome do app |
| `app_release_version` | string | `DGO Android - 4.85.1` | Versão do app |
| `player` | string | `Media3ExoPlayer` | Player utilizado |
| `player_version` | string | `1.9.0` | Versão do player |
| `plugin_version` | string | `7.3.33-Media3ExoPlayer-Android` | Versão do plugin NPAW |
| `lib_version` | string | `7.3.33-android-sdk` | Versão da lib |

### 2.6 Rede e CDN
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `cdn` | string | `Active Switching` | CDN utilizada |
| `isp` | string | `Starlink` | Provedor de internet |
| `asn` | string | `14593` | ASN do ISP |
| `connection_type` | string | `Wifi` | Tipo de conexão |
| `streaming_protocol` | string | `MPEG-DASH` | Protocolo de streaming |
| `media_resource` | string | | URL do recurso |
| `media_resource_domain` | string | | Domínio do recurso |
| `drm` | string | | DRM utilizado |
| `audio_codec` | string | `mp4a.40.2` | Codec de áudio |
| `video_codec` | string | | Codec de vídeo |
| `container_format` | string | | Formato container |

### 2.7 Erros
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `error_code` | string | `Bus: Authorization NIP` | Código do erro |
| `error_message` | string | `[Android][Bus: Auth...] - statusCode: 422` | Mensagem completa |
| `error_type` | string | `Play Failure` | Tipo de erro |
| `error_severity` | string | | Severidade |
| `error_nice_code` | string | `848660340` | Código amigável |
| `error_nice_label` | string | `ERROR` | Label do erro |
| `error_player_code` | string | `Bus: Authorization NIP` | Código do player |
| `error_player_desc` | string | `[Android][Bus: Auth...]...` | Descrição do player |
| `error_metadata` | string | | Metadata do erro |
| `in_stream_error_count` | int | `0` | Erros durante stream |
| `last_in_stream_error_code` | int | `0` | Último erro in-stream |

### 2.8 Status e Score de Qualidade
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `playback_status` | string | `On Stop`, `On Error` | Status final do playback |
| `view_status` | string | `Healthy plays`, `Plays with Startup Errors` | Status da visualização |
| `exit_status` | string | `Stopped` | Status de saída |
| `exit_type_crash` | string | `Startup Error Crash` | Tipo de crash na saída |
| `happiness_score` | float | `9.99`, `0.0` | Score de satisfação (0-10) |
| `happiness_score_label` | string | `happy`, `angry` | Label de satisfação |
| `custom_happiness_score` | float | `9.999994` | Score customizado |
| `npaw_happiness_score` | float | `-1.0` | Score NPAW |

### 2.9 Ads (Anúncios)
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `ads_count` | int | `0` | Total de ads exibidos |
| `ads_impressed_count` | int | `0` | Ads impressos |
| `ads_breaks_count` | int | `0` | Blocos de ads |
| `ads_breaks_duration` | int | `0` | Duração dos breaks |
| `ads_error_count` | int | `0` | Erros em ads |
| `time_between_ads` | int | `0` | Tempo entre ads |
| `exit_ad_status` | string | | Status de saída do ad |

### 2.10 Switches e Mudanças
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `switch_events_count` | int | `0` | Total de switches |
| `switch_events_sent_count` | int | `0` | Switches enviados |
| `switch_count_resolution` | int | `0` | Mudanças de resolução |
| `switch_count_rendition` | int | `0` | Mudanças de rendition |
| `switch_count_renditionbitrate` | int | `0` | Mudanças de bitrate |
| `switch_count_title2` | int | `0` | Mudanças de título |

### 2.11 Contexto e Dimensões Extras
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `region` | string | `LATAM` | Região |
| `viewing_mode` | string | `Streaming` | Modo de visualização |
| `offline_view` | string | `Streaming` | Se é offline |
| `nav_context` | string | `Default` | Contexto de navegação |
| `page` | string | | Página de origem |
| `referral` | string | | Referral |
| `referral_type` | string | | Tipo de referral |
| `route` | string | | Rota |
| `domain` | string | | Domínio |
| `contracted_resolution` | string | | Resolução contratada |

### 2.12 Extra Params (Customizados)
| Campo | Tipo | Exemplo | Descrição |
|-------|------|---------|-----------|
| `extraparam1` | string | `sportslander-deportesargentinaherodefault` | Contexto de navegação/origem |
| `extraparam2` | string | `Not Available` | Param customizado 2 |
| `extraparam3` | string | `OTT-D2C` | Modelo de negócio |
| `extraparam4` | string | `CH0100000000021` | ID do canal |
| `extraparam5` | string | `0` | Param customizado 5 |
| `extraparam6` | string | `Fútbol` | Categoria/gênero |
| `extraparam9` | string | `TVY7` | Rating/classificação |
| `extraparam11` | string | `SSLA-OTT` | Plataforma |
| `extraparam13` | string | `68adc6fc-7ca1-529b-...` | User ID (redundante) |

---

## 3. Exemplo Real: Análise do Usuário `68adc6fc-7ca1-529b-aaf8-c614e8f1cd32`

### 3.1 Visão Geral (últimos 6 meses: Abril-Julho 2026)

| Métrica | Valor |
|---------|-------|
| Total de sessões (amostra de 100) | 100 |
| Sessões com erro | 33 (33%) |
| Tempo total assistido | ~61.9 horas |
| Primeiro acesso no período | 28/04/2026 |
| Último acesso | 31/07/2026 |
| Dispositivo principal | UnionTV (Smart TV) — 77 sessões |
| Dispositivo secundário | Samsung SM-A045M (celular) — 23 sessões |
| App | DGO Android |
| ISP | Starlink |
| Região | LATAM |

### 3.2 Canais Assistidos (por tempo)

| Canal | Tempo Assistido | Sessões | % do Tempo |
|-------|----------------|---------|------------|
| Telefe | 26.16h | 26 | 42.3% |
| ESPN Argentina | 8.24h | 9 | 13.3% |
| TNT Sports Argentina | 6.73h | 8 | 10.9% |
| ESPN Premium ARG | 6.52h | 7 | 10.5% |
| DSPORTS (ARG) | 5.97h | 27 | 9.6% |
| Fox Sports Argentina | 4.56h | 11 | 7.4% |
| El Trece | 1.24h | 1 | 2.0% |
| Discovery ID | 0.93h | 1 | 1.5% |
| Fox Sports 2 Argentina | 0.87h | 2 | 1.4% |
| TyC Sports | 0.36h | 1 | 0.6% |
| Gran Hermano | 0.28h | 1 | 0.5% |

### 3.3 Erros Encontrados

| Tipo de Erro | Ocorrências | Descrição |
|-------------|-------------|-----------|
| Tech: Device Network | 17x | Problemas de rede no dispositivo |
| Tech: CDN | 10x | Falha na CDN/servidor de conteúdo |
| Bus: Authorization NIP | 4x | Conteúdo não autorizado (fora do pacote) |
| Tech: Unexpected | 2x | Erros inesperados |

### 3.4 Tipos de Conteúdo

| Tipo | Sessões | % |
|------|---------|---|
| EPISODE | 51 | 51% |
| SHOW | 17 | 17% |
| LIVE | 17 | 17% |
| SPORT | 15 | 15% |

### 3.5 Perfil do Usuário (Insights)

- **Perfil misto**: Entretenimento (Telefe/Gran Hermano) + Esportes (ESPN, Fox Sports, DSPORTS)
- **Heavy user de esportes argentinos**: Futebol é o gênero predominante
- **Experiência degradada**: 33% das sessões com erro — risco de churn
- **Problemas de rede**: ISP Starlink com instabilidade (17 erros de Device Network)
- **Multi-device**: Assiste principalmente na TV, usa celular como complemento
- **Tentativas frustradas**: 27 sessões DSPORTS com apenas 5.97h efetivas (muitos erros de auth)

---

## 4. Campos Mais Relevantes para Predição de Churn

Com base nos dados disponíveis, as features mais relevantes para um modelo de churn seriam:

### 4.1 Engajamento
- `effective_time` — tempo efetivo total assistido
- Frequência de sessões (contagem por período)
- Diversidade de canais (`content_channel` distintos)
- Tipos de conteúdo preferidos (`content_type`)
- Horários de uso (extrair de `init_time`)

### 4.2 Qualidade de Experiência (QoE)
- `happiness_score` — score de satisfação por sessão
- `buffer_ratio` — taxa de rebuffering
- `error_code` — presença e frequência de erros
- `startup_time` — tempo de carregamento
- `avg_bitrate` — qualidade média de vídeo

### 4.3 Problemas Técnicos
- Taxa de erro por sessão (`error_code != ""`)
- Tipos de erro predominantes
- `playback_status` = "On Error" — sessões que falharam
- `view_status` = "Plays with Startup Errors" — falhas no início

### 4.4 Dispositivo e Contexto
- `device.device_type` — tipo de dispositivo
- `connection_type` — WiFi vs Mobile
- `isp` — provedor de internet
- `app_release_version` — versão do app (desatualizado?)

### 4.5 Comportamento
- `pause_count` — quantidade de pausas
- `seek_count` — quantidade de seeks (pular conteúdo)
- `switch_events_count` — mudanças de qualidade
- Padrão de horários e dias da semana

---

## 5. Endpoints Complementares

Além do rawdata, existem outros endpoints úteis:

| Endpoint | Descrição | Uso para Churn |
|----------|-----------|----------------|
| `/{accountCode}/rawdata` | Sessões individuais | Dados brutos por user |
| `/{accountCode}/rawdata/events` | Eventos de sessão | Erros detalhados |
| `/{accountCode}/data` | Dados agregados/charts | Métricas agregadas |
| `/{accountCode}/rawdata/generic` | Dados genéricos | Consultas flexíveis |
| `/{accountCode}/rawdata/genericalerts` | Alertas | Alertas por usuário |

---

## 6. Limitações e Observações

1. **Limite por request**: Máximo de 500 registros por chamada (usar `offset` para paginar)
2. **Retenção de dados**: A API pode ter limite de retenção (verificar com NPAW)
3. **Timeout**: Consultas com limit alto (>100) e período longo podem demorar
4. **Rate limiting**: Não documentado explicitamente, usar com moderação
5. **Filtro `user_id`**: O campo correto no filtro é `user_id` (não `user`)
6. **Dimensão `user` no orderBy**: Para ordenação, usa-se `user` (sem underscore)
7. **O primeiro user testado (`9e527027-b330-5d8f-aaa2-bf6653bd6eeca`) não retornou dados** — pode ter sido deletado, ser de outro período, ou pertencer a outra conta

---

## 7. Código Python de Referência

```python
import requests
import urllib.parse
import json
from typing import Optional

class NPAWClient:
    """Cliente para a API NPAW - consulta de dados por usuário."""
    
    BASE_URL = "https://api.npaw.com"
    
    def __init__(self, account_code: str, api_key: str):
        self.account_code = account_code
        self.headers = {"npaw-api-key": api_key}
    
    def get_user_sessions(
        self,
        user_id: str,
        from_date: str = "last30days",
        to_date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        content_type: Optional[str] = None,
    ) -> dict:
        """
        Busca sessões de um usuário específico.
        
        Args:
            user_id: ID do usuário
            from_date: Data início (ex: 'last30days', 'last6months', '2026-01-01')
            to_date: Data fim (opcional)
            limit: Máximo de registros (max 500)
            offset: Offset para paginação
            content_type: Filtrar por tipo ('ALL', 'VOD', 'LIVE')
        
        Returns:
            dict com dados das sessões
        """
        filter_json = json.dumps([{
            "name": "user_filter",
            "rules": {"user_id": [user_id]}
        }])
        
        params = {
            "fromDate": from_date,
            "filter": filter_json,
            "limit": limit,
            "offset": offset,
            "orderBy": "end_at",
            "orderDirection": "desc",
        }
        
        if to_date:
            params["toDate"] = to_date
        if content_type:
            params["type"] = content_type
        
        url = f"{self.BASE_URL}/{self.account_code}/rawdata"
        response = requests.get(url, headers=self.headers, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    
    def get_all_user_sessions(
        self,
        user_id: str,
        from_date: str = "last6months",
        batch_size: int = 100,
    ) -> list:
        """
        Busca TODAS as sessões de um usuário (com paginação automática).
        
        Args:
            user_id: ID do usuário
            from_date: Data início
            batch_size: Tamanho de cada batch
        
        Returns:
            Lista com todas as sessões
        """
        all_sessions = []
        offset = 0
        
        while True:
            result = self.get_user_sessions(
                user_id=user_id,
                from_date=from_date,
                limit=batch_size,
                offset=offset,
            )
            
            if not result.get("data") or not result["data"][0].get("values"):
                break
            
            sessions = result["data"][0]["values"]
            all_sessions.extend(sessions)
            
            if len(sessions) < batch_size:
                break  # Última página
            
            offset += batch_size
        
        return all_sessions


# Exemplo de uso:
if __name__ == "__main__":
    client = NPAWClient(
        account_code="sky_brazil",
        api_key="SUA_API_KEY_AQUI"
    )
    
    # Buscar sessões de um usuário
    sessions = client.get_all_user_sessions(
        user_id="68adc6fc-7ca1-529b-aaf8-c614e8f1cd32",
        from_date="last6months"
    )
    
    print(f"Total de sessões: {len(sessions)}")
    
    # Calcular métricas
    total_time = sum(s.get("effective_time", 0) for s in sessions)
    errors = sum(1 for s in sessions if s.get("error_code"))
    channels = set(s.get("content_channel") for s in sessions if s.get("content_channel"))
    
    print(f"Tempo total: {total_time / 3600000:.1f} horas")
    print(f"Sessões com erro: {errors} ({errors/len(sessions)*100:.0f}%)")
    print(f"Canais distintos: {len(channels)}")
```

---

## 8. Resumo Executivo

| Aspecto | Status | Detalhe |
|---------|--------|---------|
| ✅ Sessões por user_id | Funciona | Endpoint rawdata com filtro user_id |
| ✅ Canais assistidos | Funciona | Campo `content_channel` |
| ✅ Tempo assistido | Funciona | Campo `effective_time` (ms) |
| ✅ Erros por sessão | Funciona | Campos `error_code`, `error_type`, `error_message` |
| ✅ Dispositivos | Funciona | Objeto `device` com modelo, OS, tipo |
| ✅ Qualidade (QoE) | Funciona | Bitrate, buffer ratio, FPS, latência |
| ✅ Score de satisfação | Funciona | `happiness_score` (0-10) |
| ✅ Tipo de conteúdo | Funciona | `content_type` (EPISODE, SPORT, LIVE, SHOW) |
| ✅ ISP/Conexão | Funciona | `isp`, `connection_type`, `asn` |
| ✅ App/Player | Funciona | Versão do app, player, plugin |
| ✅ Anúncios | Funciona | Contagem de ads, tempo, erros |
| ✅ Geolocalização | Funciona | IP, região |
| ⚠️ Histórico completo | Parcial | Limitado pela retenção de dados da NPAW |

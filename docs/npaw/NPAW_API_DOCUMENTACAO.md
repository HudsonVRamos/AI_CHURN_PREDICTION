# Documentação das APIs NPAW

> Mapeamento completo das APIs disponíveis na documentação NPAW (Integration Docs).  
> Fonte: HTMLs exportados de https://documentation.npaw.com/integration-docs/reference/  
> Base URL: `https://documentation.npaw.com/integration-docs/reference/`

---

## 1. Visão Geral das APIs

A NPAW Suite possui dois componentes principais:
- **Data Collection** — coleta de dados de players/dispositivos e envio para os servidores NPAW
- **Data Analytics** — recuperação programática dos dados via REST API (mesmo dados disponíveis na UI)

Todas as APIs retornam JSON e utilizam HTTP standard (códigos de status padrão).

---

## 2. APIs Documentadas

### 2.1 Data Collection and Analytics APIs (Geral)

| Arquivo | Descrição |
|---------|-----------|
| `about-data-collection-api.html` | Visão geral do sistema de Data Collection e Data Analytics. Explica a arquitetura: plugin coleta dados do player/dispositivo e envia para os servidores NPAW 24/7. |
| `data.html` | Mesmo conteúdo (Data Collection and Analytics APIs) |

---

### 2.2 Admin API (Gerenciamento de Contas e Usuários)

API REST para gerenciar contas, usuários, roles e permissões da NPAW Suite programaticamente.

| Arquivo | Título | Método | Descrição |
|---------|--------|--------|-----------|
| `about-admin-api.html` | About Admin API | — | Visão geral: criar, atualizar, deletar e gerenciar contas de usuário |
| **Contas** | | | |
| `get_open-accounts.html` | Get accounts | GET | Listar contas |
| `post_open-accounts.html` | Clone account | POST | Clonar uma conta |
| `put_open-accounts.html` | Update account | PUT | Atualizar conta |
| `delete_open-accounts.html` | Delete account | DELETE | Deletar conta |
| **Usuários** | | | |
| `get_open-users.html` | Get user | GET | Obter dados de usuário |
| `post_open-users.html` | Create user | POST | Criar usuário |
| `put_open-users.html` | Update user | PUT | Atualizar usuário |
| `delete_open-users.html` | Delete user | DELETE | Deletar usuário |
| `users.html` | Get user | GET | Obter dados de usuário (duplicata) |
| **Associação Usuário-Conta** | | | |
| `get_open-users-accounts.html` | Get users accounts | GET | Listar associações usuário-conta |
| `post_open-users-accounts.html` | Create user account | POST | Criar associação usuário-conta |
| `delete_open-users-accounts.html` | Delete user account | DELETE | Deletar associação usuário-conta |
| **Roles (Funções)** | | | |
| `get_open-roles.html` | Get roles | GET | Listar roles |
| `post_open-roles.html` | Create role | POST | Criar role |
| `put_open-roles.html` | Update role | PUT | Atualizar role |
| `delete_open-roles.html` | Delete role | DELETE | Deletar role |
| **Permissões de Roles** | | | |
| `get_open-roles-permissions.html` | Get all roles permissions | GET | Listar permissões por role |
| `post_open-roles-permissions.html` | Create roles permissions | POST | Criar permissões para role |
| `put_open-roles-permissions.html` | Update roles permissions | PUT | Atualizar permissões de role |
| `delete_open-roles-permissions.html` | Delete roles permissions | DELETE | Deletar permissões de role |
| **Associação Usuário-Role** | | | |
| `get_open-user-roles.html` | Get all user roles | GET | Listar roles de um usuário |
| `post_open-user-roles.html` | Create user roles | POST | Associar role a usuário |
| `delete_open-user-roles.html` | Delete user roles | DELETE | Remover role de usuário |
| **GDPR** | | | |
| `get_open-gdpr.html` | Get GDPR rules | GET | Listar regras GDPR |
| `post_open-gdpr.html` | Create GDPR rule | POST | Criar regra GDPR |
| `delete_open-gdpr.html` | Delete GDPR rule | DELETE | Deletar regra GDPR |

---

### 2.3 Data Analytics API (Consulta de Dados)

API REST para recuperar dados analíticos (mesmos dados disponíveis na UI da NPAW).

| Arquivo | Título | Descrição |
|---------|--------|-----------|
| `about-analytics-data-collection-api.html` | About Data Analytics API | Visão geral: recuperar via HTTP todos os dados mostrados na UI |
| `analytics.html` | About Data Analytics API | Mesmo conteúdo |
| `get_accountcode-data.html` | Charts data | Obter dados para gráficos/charts |
| `get_accountcode-rawdata.html` | Raw session data | Dados brutos de sessão |
| `get_accountcode-rawdata-events.html` | Events request | Consultar eventos |
| `get_accountcode-rawdata-generic.html` | Generic data retrieval | Consulta genérica de dados |
| `get_accountcode-rawdata-genericalerts.html` | Alerts retrieval | Consultar alertas |
| `get_accountcode-rawdata-switching.html` | Switching events request | Eventos de switching/troca |
| `get_data.html` | /data | Endpoint /data |
| `dimensions.html` | Dimensions | Dimensões disponíveis para consultas |
| `event-types.html` | Event types | Tipos de eventos disponíveis |
| `annotations.html` | Annotations | Anotações |

---

### 2.4 Video Data Collection API (Coleta de Dados de Vídeo)

API para coleta de dados de streaming de vídeo. Rastreia eventos de playback (loading, buffering, erros, etc.).

| Arquivo | Título | Método | Descrição |
|---------|--------|--------|-----------|
| `about-video-data-collection-api.html` | About Video Data Collection API | — | Visão geral: análise de experiências de streaming de vídeo |
| `get_infinity-video-start-1.html` | Video requested | GET | Vídeo solicitado/requisitado |
| `get_infinity-video-jointime-1.html` | Video playback started | GET | Playback iniciou (join time) |
| `get_infinity-video-ping-1.html` | Video playback heartbeat | GET | Heartbeat do playback |
| `get_infinity-video-pause-1.html` | Video is paused | GET | Vídeo pausado |
| `get_infinity-video-resume-1.html` | Video is resumed after a pause | GET | Vídeo resumido após pausa |
| `get_infinity-video-seek-1.html` | Jumping to a different section | GET | Seek/salto para outra posição |
| `get_infinity-video-bufferunderrun-1.html` | Video rebuffering | GET | Rebuffering do vídeo |
| `get_infinity-video-error-1.html` | Video error | GET | Erro no vídeo |
| `get_infinity-video-event-1.html` | Send a custom event | GET | Evento customizado de vídeo |
| `get_infinity-video-stop-1.html` | End of content / user ended | GET | Fim do conteúdo ou usuário encerrou |

---

### 2.5 Ads Data Collection API (Coleta de Dados de Anúncios)

API para coleta de dados de anúncios em vídeo. Funciona em conjunto com a Video Data Collection API.

| Arquivo | Título | Método | Descrição |
|---------|--------|--------|-----------|
| `about-ads-data-collection.html` | About Ads Data Collection API | — | Visão geral: coleta de analytics de ads |
| `get_infinity-video-adbreakstart-1.html` | Start of an ad break | GET | Início de um bloco de anúncios |
| `get_infinity-video-adbreakstop-1.html` | End of an ad break | GET | Fim de um bloco de anúncios |
| `get_infinity-video-admanifest-1.html` | Ad configuration setup | GET | Configuração de ad dentro de sessão de vídeo |
| `get_infinity-video-adstart-1.html` | Ad creativity requested | GET | Criativo de ad solicitado |
| `get_infinity-video-adjoin-1.html` | Ad creativity started | GET | Criativo de ad iniciou |
| `get_infinity-video-adpause-1.html` | Ad creativity is paused | GET | Criativo de ad pausado |
| `get_infinity-video-adresume-1.html` | Ad creativity resumed | GET | Criativo de ad resumido |
| `get_infinity-video-adbuffer-1.html` | Ad creativity rebuffering | GET | Rebuffering do criativo de ad |
| `get_infinity-video-adclick-1.html` | Ad clicked | GET | Clique no anúncio |
| `get_infinity-video-aderror-1.html` | Ad error | GET | Erro no anúncio |
| `get_infinity-video-adquartile-1.html` | Ad quartile reached | GET | Quartil do ad atingido |
| `get_infinity-video-adstop-1.html` | End of an ad creativity | GET | Fim do criativo de ad |

---

### 2.6 Session Data Collection API (Coleta de Dados de Sessão)

API para coleta de dados de sessão da aplicação (engagement e comportamento do usuário).

| Arquivo | Título | Método | Descrição |
|---------|--------|--------|-----------|
| `about-session-data-collection-api.html` | About Session Data Collection API | — | Visão geral: dados de início/fim de sessão, heartbeat, navegação |
| `get_infinity-session-start-1.html` | Start a new session | GET | Iniciar nova sessão |
| `get_infinity-session-stop-1.html` | Stop current session | GET | Encerrar sessão atual |
| `get_infinity-session-beat-1.html` | Session heartbeat | GET | Heartbeat da sessão |
| `get_infinity-session-nav-1.html` | Navigate to a new page | GET | Navegação para nova página |
| `get_infinity-session-event-1.html` | Send a custom event | GET | Enviar evento customizado |

---

### 2.7 CDN Balancer API (Gerenciamento de CDN)

API para gerenciamento e recomendação de CDNs.

| Arquivo | Título | Método | Descrição |
|---------|--------|--------|-----------|
| `cdn-management.html` | CDN Management | — | Visão geral do gerenciamento de CDN |
| `cdn-recommender.html` | Recommended CDNs Using Default Profile | — | Recomendações de CDN com perfil padrão |
| **Perfis** | | | |
| `get_cdnbalancer-account-profiles-1.html` | Get All Profiles | GET | Listar todos os perfis de CDN |
| `post_cdnbalancer-account-configuration-create-1.html` | Create New Profile | POST | Criar novo perfil |
| `post_cdnbalancer-account-profile-1.html` | Modify Existing Profile | POST | Modificar perfil existente |
| `get_cdnbalancer-account-configuration-delete-profileid-1.html` | Delete Existing Profile | GET | Deletar perfil |
| **Decisão/Recomendação** | | | |
| `get_accountcode-decision.html` | Recommended CDNs Using Default Profile | GET | CDNs recomendadas (perfil padrão) |
| `get_accountcode-profile-decision.html` | Recommended CDNs of a Given Profile | GET | CDNs recomendadas (perfil específico) |
| **Redirect HTTP** | | | |
| `get_accountcode-redir-1.html` | HTTP Redirect Using Default Profile | GET | Redirect HTTP (perfil padrão) |
| `get_accountcode-profile-redir-1.html` | HTTP Redirect Using Specific Profile | GET | Redirect HTTP (perfil específico) |
| `get_resource-path-1.html` | Subdomain-based HTTP Redirect | GET | Redirect HTTP por subdomínio |
| **Informação CDN (HEAD)** | | | |
| `head_accountcode-redir-1.html` | Get CDN Info Using Default Profile (HEAD) | HEAD | Info CDN via HEAD (perfil padrão) |
| `head_accountcode-profile-redir-1.html` | Get CDN Info Using Specific Profile (HEAD) | HEAD | Info CDN via HEAD (perfil específico) |
| `head_resource-path-1.html` | Get CDN Info via Subdomain (HEAD) | HEAD | Info CDN via subdomínio (HEAD) |

---

### 2.8 Autenticação e Guides

| Arquivo | Título | Descrição |
|---------|--------|-----------|
| `authentication-methods.html` | Authentication Methods | Métodos de autenticação disponíveis |
| `authentication-methods-copy.html` | Authentication Methods | Cópia/backup da página de autenticação |
| `how-to-generate-an-auth-token.html` | Generate an Authentication Token | Como gerar token de autenticação (Analytics) |
| `how-to-generate-an-auth-token-cdn.html` | Generate an Authentication Token | Como gerar token de autenticação (CDN) |
| `how-to-generate-filters.html` | Generate Filters | Como gerar filtros para consultas |
| `generate-api-call.html` | How to Generate an API Call | Como construir uma chamada de API |

---

## 3. Resumo por Categoria

| Categoria | Qtd. Endpoints | Descrição |
|-----------|---------------|-----------|
| Admin API | 20 | CRUD de contas, usuários, roles, permissões, GDPR |
| Data Analytics API | 10 | Consulta de dados, charts, raw data, eventos, alertas |
| Video Data Collection | 11 | Eventos de playback de vídeo (start, stop, pause, seek, error, etc.) |
| Ads Data Collection | 13 | Eventos de anúncios (break start/stop, ad start/stop, quartile, etc.) |
| Session Data Collection | 5 | Eventos de sessão (start, stop, heartbeat, nav, custom event) |
| CDN Balancer | 11 | Perfis, recomendação de CDN, redirect HTTP |
| Autenticação/Guides | 6 | Tokens, filtros, como gerar chamadas |

**Total: ~76 endpoints/páginas documentadas** (91 HTMLs, com algumas duplicatas)

---

## 4. Observações

1. **Autenticação**: As APIs utilizam tokens de autenticação. Consultar `authentication-methods.html` e `how-to-generate-an-auth-token.html` para detalhes.
2. **Formato de resposta**: Todas as APIs retornam JSON.
3. **Protocolo**: REST sobre HTTP/HTTPS.
4. **Páginas protegidas**: Todas as páginas HTML foram salvas com meta description "This page is password-protected", indicando que o conteúdo detalhado (parâmetros, exemplos) requer autenticação no portal da NPAW.
5. **Arquivo PDF**: `502_introducing_lowlatency_hls.pdf` — documento sobre Low Latency HLS (não é API, é material de referência).
6. **URL base da documentação**: `https://documentation.npaw.com/integration-docs/reference/`

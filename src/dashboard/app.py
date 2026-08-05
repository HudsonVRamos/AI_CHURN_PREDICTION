"""Dashboard analítico interativo para a plataforma de predição de churn.

Aplicação Streamlit multi-página com navegação por sidebar.
Páginas: Overview, Charts, Subscriber Detail, Export.

Suporta refresh automático após nova execução do pipeline.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7
"""

from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

# Importação condicional para permitir import sem plotly instalado
try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from src.dashboard.data_source import DashboardDataSource


# --- Configuração da página ---
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Constantes ---
PAGES = ["Overview", "Charts", "Subscriber Detail", "Export"]
AUTO_REFRESH_INTERVAL_SECONDS = 60


# --- Inicialização do session_state ---
def _init_session_state() -> None:
    """Inicializa variáveis no session_state para cache entre refreshes."""
    if "data_source" not in st.session_state:
        st.session_state.data_source = DashboardDataSource()
    if "last_execution_id" not in st.session_state:
        st.session_state.last_execution_id = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None


def _load_data() -> None:
    """Carrega dados do data source e atualiza o session_state."""
    ds: DashboardDataSource = st.session_state.data_source
    execution = ds.get_latest_execution()
    st.session_state.latest_execution = execution

    # Carrega predições
    predictions = ds.get_predictions()
    st.session_state.predictions = predictions
    st.session_state.last_refresh = datetime.now().isoformat()

    if execution:
        st.session_state.last_execution_id = execution.execution_id


def _check_auto_refresh() -> None:
    """Verifica se há nova execução e recarrega dados automaticamente.

    Compara o execution_id da última execução com o armazenado no state.
    Se diferente, recarrega todos os dados (Requirement 14.6).
    """
    ds: DashboardDataSource = st.session_state.data_source
    execution = ds.get_latest_execution()

    if execution and execution.execution_id != st.session_state.last_execution_id:
        _load_data()
        st.toast("🔄 Dados atualizados — nova execução detectada!", icon="✅")


# --- Sidebar: Navegação ---
def _render_sidebar() -> str:
    """Renderiza sidebar com navegação e informações de execução."""
    with st.sidebar:
        st.title("📊 Churn Dashboard")
        st.divider()

        page = st.radio("Navegação", PAGES, index=0)

        st.divider()
        st.subheader("Última Execução")

        execution = st.session_state.get("latest_execution")
        if execution:
            st.text(f"ID: {execution.execution_id[:8]}...")
            st.text(f"Status: {execution.status}")
            st.text(f"Modo: {execution.mode}")
            st.text(f"Processados: {execution.users_processed}")
            st.text(f"Falhas: {execution.users_failed}")
        else:
            st.info("Nenhuma execução encontrada.")

        st.divider()
        last_refresh = st.session_state.get("last_refresh", "—")
        st.caption(f"Último refresh: {last_refresh}")

        if st.button("🔄 Atualizar Dados"):
            _load_data()
            st.rerun()

    return page


# --- Página: Overview (KPIs) ---
def _render_overview() -> None:
    """Página Overview com KPIs principais.

    Exibe: total analisados, contagem por risk tier, média de churn probability.
    Requirement 14.1
    """
    st.title("📋 Overview")
    st.markdown("Visão geral dos indicadores de churn prediction.")

    predictions = st.session_state.get("predictions", [])

    if not predictions:
        st.warning("Nenhuma predição disponível. Execute o pipeline primeiro.")
        return

    total = len(predictions)
    high_risk = [p for p in predictions if p.get("risk_tier") == "High"]
    medium_risk = [p for p in predictions if p.get("risk_tier") == "Medium"]
    low_risk = [p for p in predictions if p.get("risk_tier") == "Low"]
    avg_churn = (
        sum(p.get("churn_probability", 0) for p in predictions) / total
        if total > 0
        else 0.0
    )

    # KPIs em colunas
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Analisados", total)
    with col2:
        st.metric("🔴 High Risk", len(high_risk))
    with col3:
        st.metric("🟡 Medium Risk", len(medium_risk))
    with col4:
        st.metric("🟢 Low Risk", len(low_risk))
    with col5:
        st.metric("Média Churn Prob.", f"{avg_churn:.2%}")

    st.divider()

    # Tabela resumo
    st.subheader("Distribuição por Risco")
    dist_data = {
        "Nível de Risco": ["High", "Medium", "Low"],
        "Quantidade": [len(high_risk), len(medium_risk), len(low_risk)],
        "Percentual": [
            f"{len(high_risk) / total * 100:.1f}%" if total > 0 else "0%",
            f"{len(medium_risk) / total * 100:.1f}%" if total > 0 else "0%",
            f"{len(low_risk) / total * 100:.1f}%" if total > 0 else "0%",
        ],
    }
    st.table(dist_data)


# --- Página: Charts ---
def _render_charts() -> None:
    """Página Charts com gráficos interativos.

    Gráficos: distribuição de risco (pie), histograma de scores,
    top fatores (bar), evolução temporal (line).
    Requirement 14.2
    """
    st.title("📈 Charts")
    st.markdown("Visualizações interativas dos dados de predição.")

    predictions = st.session_state.get("predictions", [])

    if not predictions:
        st.warning("Nenhuma predição disponível para gerar gráficos.")
        return

    if not HAS_PLOTLY:
        st.error(
            "Biblioteca plotly não instalada. "
            "Execute: pip install plotly"
        )
        return

    # --- Filtros ---
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        risk_filter = st.selectbox(
            "Filtrar por Risco",
            ["All", "High", "Medium", "Low"],
            index=0,
        )
    with col_filter2:
        period_filter = st.selectbox(
            "Período",
            ["Todos", "Última Semana", "Último Mês", "Últimos 3 Meses"],
            index=0,
        )

    # Aplica filtros
    filtered = predictions
    if risk_filter != "All":
        filtered = [
            p for p in filtered
            if p.get("risk_tier") == risk_filter
        ]

    # Aplica filtro de período (baseado no timestamp das predições)
    if period_filter != "Todos":
        from datetime import timedelta as _td
        now = datetime.now()
        days_map = {
            "Última Semana": 7,
            "Último Mês": 30,
            "Últimos 3 Meses": 90,
        }
        days = days_map.get(period_filter, 90)
        cutoff = (now - _td(days=days)).isoformat()
        filtered = [
            p for p in filtered
            if p.get("timestamp", "") >= cutoff
        ]

    if not filtered:
        st.info("Nenhum dado após aplicar filtros.")
        return

    # --- Gráfico 1: Distribuição de Risco (Pie) ---
    st.subheader("Distribuição por Nível de Risco")
    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    for p in filtered:
        tier = p.get("risk_tier", "Low")
        if tier in risk_counts:
            risk_counts[tier] += 1

    fig_pie = px.pie(
        names=list(risk_counts.keys()),
        values=list(risk_counts.values()),
        color=list(risk_counts.keys()),
        color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
        title="Distribuição por Risco",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Gráfico 2: Histograma de Scores ---
    st.subheader("Distribuição de Churn Scores")
    scores = [p.get("churn_probability", 0) for p in filtered]
    fig_hist = px.histogram(
        x=scores,
        nbins=20,
        labels={"x": "Churn Probability", "y": "Quantidade"},
        title="Histograma de Churn Probability",
        color_discrete_sequence=["#6366F1"],
    )
    fig_hist.update_layout(xaxis_range=[0, 1])
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Gráfico 3: Top Fatores (Bar) ---
    st.subheader("Top Fatores de Influência")
    _render_top_factors_chart(filtered)

    # --- Gráfico 4: Evolução Temporal (Line) ---
    st.subheader("Evolução Temporal")
    _render_temporal_chart(filtered)


def _render_top_factors_chart(predictions: list[dict]) -> None:
    """Renderiza gráfico de barras com top fatores agregados da população."""
    # Agrega SHAP contributions de todas as predições que possuem shap_results
    factor_weights: dict[str, float] = {}
    count_with_shap = 0

    for p in predictions:
        shap = p.get("shap_results") if isinstance(p.get("shap_results"), dict) else {}
        if shap:
            count_with_shap += 1
            for feature_name, weight in shap.items():
                if isinstance(weight, (int, float)):
                    factor_weights[feature_name] = (
                        factor_weights.get(feature_name, 0.0) + abs(float(weight))
                    )

    if not factor_weights:
        st.info("Dados de explicabilidade (SHAP) não disponíveis.")
        return

    # Normaliza pela quantidade de predições
    if count_with_shap > 0:
        factor_weights = {
            k: v / count_with_shap for k, v in factor_weights.items()
        }

    # Top 10
    sorted_factors = sorted(factor_weights.items(), key=lambda x: x[1], reverse=True)[:10]
    names = [f[0] for f in sorted_factors]
    values = [f[1] for f in sorted_factors]

    fig_bar = px.bar(
        x=values,
        y=names,
        orientation="h",
        labels={"x": "Impacto Médio (|SHAP|)", "y": "Feature"},
        title="Top 10 Fatores de Influência (SHAP)",
        color_discrete_sequence=["#8B5CF6"],
    )
    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_bar, use_container_width=True)


def _render_temporal_chart(predictions: list[dict]) -> None:
    """Renderiza gráfico de linha com evolução temporal dos scores."""
    # Agrupa por timestamp (data) e calcula média de churn_probability por dia
    daily_data: dict[str, list[float]] = {}

    for p in predictions:
        ts = p.get("timestamp", "")
        if not ts:
            continue
        # Extrai apenas a data (YYYY-MM-DD)
        date_str = ts[:10] if len(ts) >= 10 else ts
        if date_str not in daily_data:
            daily_data[date_str] = []
        daily_data[date_str].append(p.get("churn_probability", 0.0))

    if not daily_data:
        st.info("Dados temporais insuficientes para gráfico de evolução.")
        return

    # Ordena por data
    sorted_dates = sorted(daily_data.keys())
    avg_scores = [
        sum(daily_data[d]) / len(daily_data[d]) for d in sorted_dates
    ]

    # Conta por risk tier por data
    daily_risk: dict[str, dict[str, int]] = {}
    for p in predictions:
        ts = p.get("timestamp", "")
        if not ts:
            continue
        date_str = ts[:10] if len(ts) >= 10 else ts
        if date_str not in daily_risk:
            daily_risk[date_str] = {"High": 0, "Medium": 0, "Low": 0}
        tier = p.get("risk_tier", "Low")
        if tier in daily_risk[date_str]:
            daily_risk[date_str][tier] += 1

    # Gráfico de evolução da média de churn probability
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=sorted_dates,
        y=avg_scores,
        mode="lines+markers",
        name="Média Churn Prob.",
        line={"color": "#EF4444"},
    ))
    fig_line.update_layout(
        title="Evolução Temporal - Média de Churn Probability",
        xaxis_title="Data",
        yaxis_title="Média Churn Probability",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Gráfico de evolução por risk tier
    if daily_risk:
        sorted_risk_dates = sorted(daily_risk.keys())
        high_counts = [daily_risk[d].get("High", 0) for d in sorted_risk_dates]
        medium_counts = [daily_risk[d].get("Medium", 0) for d in sorted_risk_dates]
        low_counts = [daily_risk[d].get("Low", 0) for d in sorted_risk_dates]

        fig_risk_line = go.Figure()
        fig_risk_line.add_trace(go.Scatter(
            x=sorted_risk_dates, y=high_counts,
            mode="lines+markers", name="High Risk",
            line={"color": "#EF4444"},
        ))
        fig_risk_line.add_trace(go.Scatter(
            x=sorted_risk_dates, y=medium_counts,
            mode="lines+markers", name="Medium Risk",
            line={"color": "#F59E0B"},
        ))
        fig_risk_line.add_trace(go.Scatter(
            x=sorted_risk_dates, y=low_counts,
            mode="lines+markers", name="Low Risk",
            line={"color": "#10B981"},
        ))
        fig_risk_line.update_layout(
            title="Evolução Temporal - Assinantes por Nível de Risco",
            xaxis_title="Data",
            yaxis_title="Quantidade",
        )
        st.plotly_chart(fig_risk_line, use_container_width=True)


# --- Página: Subscriber Detail ---
def _render_subscriber_detail() -> None:
    """Página de detalhe individual de assinante.

    Busca por user_id e exibe score, confidence, top features, explanation.
    Requirement 14.3, 14.4
    """
    st.title("🔍 Subscriber Detail")
    st.markdown("Busque por User ID para ver detalhes da predição.")

    user_id = st.text_input(
        "User ID",
        placeholder="Ex: 550e8400-e29b-41d4-a716-446655440000",
    )

    if not user_id:
        st.info("Digite um User ID para buscar.")
        return

    if st.button("Buscar") or user_id:
        ds: DashboardDataSource = st.session_state.data_source
        detail = ds.get_subscriber_detail(user_id.strip())

        if not detail:
            st.error(f"Nenhuma predição encontrada para o User ID: {user_id}")
            return

        # Informações principais
        st.subheader("Resultado da Predição")
        col1, col2, col3, col4 = st.columns(4)

        churn_prob = detail.get("churn_probability", 0.0)
        confidence = detail.get("confidence", 0.0)
        risk_tier = detail.get("risk_tier", "—")

        with col1:
            st.metric("Churn Probability", f"{churn_prob:.2%}")
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        with col3:
            risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(
                risk_tier, "⚪"
            )
            st.metric("Risk Tier", f"{risk_color} {risk_tier}")
        with col4:
            st.metric("Model Version", detail.get("model_version", "—"))

        st.caption(f"Timestamp: {detail.get('timestamp', '—')}")
        st.divider()

        # Top Features (SHAP)
        st.subheader("Top Features (SHAP)")
        shap_results = detail.get("shap_results", {})
        if shap_results and isinstance(shap_results, dict):
            sorted_features = sorted(
                shap_results.items(),
                key=lambda x: abs(float(x[1])) if isinstance(x[1], (int, float)) else 0,
                reverse=True,
            )[:10]

            feature_names = [f[0] for f in sorted_features]
            feature_values = [float(f[1]) for f in sorted_features]

            if HAS_PLOTLY:
                colors = [
                    "#EF4444" if v > 0 else "#10B981" for v in feature_values
                ]
                fig_shap = go.Figure(go.Bar(
                    x=feature_values,
                    y=feature_names,
                    orientation="h",
                    marker_color=colors,
                ))
                fig_shap.update_layout(
                    title="Contribuição de Features (SHAP Values)",
                    xaxis_title="SHAP Value",
                    yaxis={"categoryorder": "total ascending"},
                    height=400,
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                for name, value in sorted_features:
                    direction = "↑ churn" if value > 0 else "↓ churn"
                    st.text(f"  {name}: {value:.4f} ({direction})")
        else:
            st.info("Dados SHAP não disponíveis para este assinante.")

        st.divider()

        # Explicação Bedrock
        st.subheader("Explicação (AWS Bedrock)")
        explanation = detail.get("bedrock_explanation")
        explanation_status = detail.get("explanation_status", "unavailable")

        if explanation:
            st.markdown(explanation)
        elif explanation_status == "pending":
            st.warning("Explicação em processamento...")
        else:
            st.info("Explicação em linguagem natural não disponível.")

        st.divider()

        # Features do assinante
        st.subheader("Features Comportamentais")
        features = detail.get("features")
        if features and isinstance(features, dict):
            # Exibe em formato de tabela
            feature_data = {
                "Feature": list(features.keys()),
                "Valor": [f"{v:.4f}" if isinstance(v, float) else str(v) for v in features.values()],
            }
            st.table(feature_data)
        else:
            st.info("Features não disponíveis para este assinante.")

        # Histórico temporal
        st.divider()
        st.subheader("Histórico de Predições")
        history = ds.get_history(user_id.strip())
        if history:
            timestamps = [h.get("timestamp", "")[:10] for h in history]
            probs = [h.get("churn_probability", 0.0) for h in history]

            if HAS_PLOTLY and len(history) > 1:
                fig_hist_line = px.line(
                    x=timestamps,
                    y=probs,
                    labels={"x": "Data", "y": "Churn Probability"},
                    title=f"Evolução do Score — {user_id[:8]}...",
                    markers=True,
                )
                fig_hist_line.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig_hist_line, use_container_width=True)
            else:
                for h in history:
                    st.text(
                        f"  {h.get('timestamp', '—')}: "
                        f"{h.get('churn_probability', 0):.2%} "
                        f"({h.get('risk_tier', '—')})"
                    )
        else:
            st.info("Nenhum histórico disponível.")


# --- Página: Export ---
def _render_export() -> None:
    """Página de exportação de dados filtrados.

    Permite download em JSON e Markdown com filtros aplicados.
    Requirement 14.5
    """
    st.title("📥 Export")
    st.markdown("Exporte dados filtrados em formato JSON ou Markdown.")

    predictions = st.session_state.get("predictions", [])

    if not predictions:
        st.warning("Nenhuma predição disponível para exportar.")
        return

    # Filtros
    st.subheader("Filtros")
    col1, col2 = st.columns(2)

    with col1:
        risk_filter = st.selectbox(
            "Nível de Risco",
            ["All", "High", "Medium", "Low"],
            index=0,
            key="export_risk_filter",
        )
    with col2:
        min_prob = st.slider(
            "Churn Probability Mínima",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="export_min_prob",
        )

    # Aplica filtros
    filtered = predictions
    if risk_filter != "All":
        filtered = [p for p in filtered if p.get("risk_tier") == risk_filter]
    if min_prob > 0:
        filtered = [
            p for p in filtered if p.get("churn_probability", 0) >= min_prob
        ]

    st.info(f"**{len(filtered)}** registros após filtros (de {len(predictions)} total).")

    if not filtered:
        st.warning("Nenhum registro corresponde aos filtros selecionados.")
        return

    st.divider()
    st.subheader("Download")

    col_dl1, col_dl2 = st.columns(2)

    # Download JSON
    with col_dl1:
        json_data = json.dumps(filtered, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    # Download Markdown
    with col_dl2:
        md_content = _generate_markdown_report(filtered)
        st.download_button(
            label="📝 Download Markdown",
            data=md_content,
            file_name=f"churn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

    # Preview
    st.divider()
    st.subheader("Preview dos Dados")
    preview_data = []
    for p in filtered[:20]:
        preview_data.append({
            "User ID": p.get("user_id", "—")[:12] + "...",
            "Churn Prob.": f"{p.get('churn_probability', 0):.2%}",
            "Confidence": f"{p.get('confidence', 0):.2%}",
            "Risk": p.get("risk_tier", "—"),
            "Timestamp": p.get("timestamp", "—")[:10],
        })

    if preview_data:
        st.table(preview_data)
        if len(filtered) > 20:
            st.caption(f"Mostrando 20 de {len(filtered)} registros.")


def _generate_markdown_report(predictions: list[dict]) -> str:
    """Gera relatório em formato Markdown a partir das predições filtradas."""
    lines = [
        "# Relatório de Predição de Churn",
        "",
        f"**Data de Geração:** {datetime.now().isoformat()}",
        f"**Total de Assinantes:** {len(predictions)}",
        "",
    ]

    # Resumo por risco
    high = sum(1 for p in predictions if p.get("risk_tier") == "High")
    medium = sum(1 for p in predictions if p.get("risk_tier") == "Medium")
    low = sum(1 for p in predictions if p.get("risk_tier") == "Low")
    avg_prob = (
        sum(p.get("churn_probability", 0) for p in predictions) / len(predictions)
        if predictions
        else 0
    )

    lines.extend([
        "## Resumo",
        "",
        "| Indicador | Valor |",
        "|-----------|-------|",
        f"| High Risk | {high} |",
        f"| Medium Risk | {medium} |",
        f"| Low Risk | {low} |",
        f"| Média Churn Prob. | {avg_prob:.2%} |",
        "",
        "## Detalhes por Assinante",
        "",
        "| User ID | Churn Prob. | Confidence | Risk | Timestamp |",
        "|---------|-------------|------------|------|-----------|",
    ])

    for p in predictions:
        uid = p.get("user_id", "—")
        prob = f"{p.get('churn_probability', 0):.2%}"
        conf = f"{p.get('confidence', 0):.2%}"
        risk = p.get("risk_tier", "—")
        ts = p.get("timestamp", "—")[:10]
        lines.append(f"| {uid} | {prob} | {conf} | {risk} | {ts} |")

    lines.append("")
    return "\n".join(lines)


# --- Main ---
def main() -> None:
    """Função principal do dashboard Streamlit."""
    _init_session_state()

    # Carrega dados na primeira execução
    if st.session_state.get("last_refresh") is None:
        _load_data()

    # Verifica auto-refresh (nova execução do pipeline)
    _check_auto_refresh()

    # Navegação
    page = _render_sidebar()

    # Renderiza página selecionada
    if page == "Overview":
        _render_overview()
    elif page == "Charts":
        _render_charts()
    elif page == "Subscriber Detail":
        _render_subscriber_detail()
    elif page == "Export":
        _render_export()


if __name__ == "__main__":
    main()

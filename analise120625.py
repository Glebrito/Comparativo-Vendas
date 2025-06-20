import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
import os
import numpy as np

# Importações condicionais para PDF
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_TABELA_DISPONIVEL = True
except ImportError as e:
    PDF_TABELA_DISPONIVEL = False

# Configurações do Google Sheets
SHEET_ID = "1qRAOnH7bKsUEYr9zHGznP_6NflusS-IHbDBSlDgu0I8"
SHEET_NAME = "Dados"
CREDENTIALS_PATH = "C://Users/glebr/Downloads/bustling-day-459711-q8-e889589cda14.json"  # Caminho para o arquivo de credenciais local

# Configuração da página
st.set_page_config(layout="wide", page_title="Análise Comercial")

def formatar_real(valor):
    """Formata valor para Real brasileiro"""
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def formatar_porcentagem(valor):
    """Formata valor para porcentagem"""
    return f"{valor:.2f}%"

def formatar_status_csv(valor):
    """Formata status para CSV com texto descritivo"""
    if isinstance(valor, str):
        if '✅' in valor:
            return "Meta Batida"
        elif '❌' in valor:
            return "Meta Não Batida"
    return str(valor)

def formatar_status_pdf(valor):
    """Formata status para PDF mantendo emojis"""
    return str(valor)

def obter_ordem_meses():
    """Retorna a ordem correta dos meses com variações de capitalização"""
    return [
        'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
        'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
    ]

def normalizar_nome_mes(mes):
    """Normaliza o nome do mês para capitalização correta"""
    mes_limpo = str(mes).strip()
    # Capitalizar primeira letra e deixar resto minúsculo
    return mes_limpo.capitalize()

def obter_indice_mes(mes):
    """Obtém o índice do mês tratando variações de capitalização"""
    try:
        mes_normalizado = normalizar_nome_mes(mes)
        ordem_meses = obter_ordem_meses()
        return ordem_meses.index(mes_normalizado)
    except ValueError:
        # Se não encontrar, tentar algumas variações comuns
        mes_lower = mes.lower()
        mapeamento_meses = {
            'janeiro': 0, 'jan': 0,
            'fevereiro': 1, 'fev': 1,
            'março': 2, 'mar': 2, 'marco': 2,
            'abril': 3, 'abr': 3,
            'maio': 4, 'mai': 4,
            'junho': 5, 'jun': 5,
            'julho': 6, 'jul': 6,
            'agosto': 7, 'ago': 7,
            'setembro': 8, 'set': 8,
            'outubro': 9, 'out': 9,
            'novembro': 10, 'nov': 10,
            'dezembro': 11, 'dez': 11
        }
        
        if mes_lower in mapeamento_meses:
            return mapeamento_meses[mes_lower]
        else:
            st.error(f"Mês não reconhecido: {mes}")
            return 0

def criar_pdf_tabela(df, titulo, nome_arquivo):
    """Cria PDF de uma tabela"""
    if not PDF_TABELA_DISPONIVEL:
        st.error("PDF de tabelas não disponível. Instale: pip install reportlab")
        return None
        
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph(titulo, title_style))
        story.append(Spacer(1, 20))
        
        # Preparar dados da tabela para PDF (mantendo emojis)
        df_pdf = df.copy()
        if 'Status' in df_pdf.columns:
            df_pdf['Status'] = df_pdf['Status'].apply(formatar_status_pdf)
        
        data = [df_pdf.columns.tolist()]  # Header
        for _, row in df_pdf.iterrows():
            data.append([str(cell) for cell in row.tolist()])
        
        # Criar tabela
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        
        # Data de geração
        story.append(Spacer(1, 20))
        data_geracao = f"Relatório gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        story.append(Paragraph(data_geracao, styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Erro ao gerar PDF da tabela: {str(e)}")
        return None

def botao_download_pdf_tabela(df, titulo, nome_arquivo):
    """Cria botão de download para PDF da tabela"""
    if not PDF_TABELA_DISPONIVEL:
        st.info("📄 PDF não disponível - instale: pip install reportlab")
        return
        
    pdf_bytes = criar_pdf_tabela(df, titulo, nome_arquivo)
    if pdf_bytes:
        st.download_button(
            label="📄 Download PDF",
            data=pdf_bytes,
            file_name=f"{nome_arquivo}.pdf",
            mime="application/pdf",
            key=f"pdf_{nome_arquivo}_{hash(str(df.values.tolist()))}"
        )

def botao_download_csv_tabela(df, nome_arquivo):
    """Cria botão de download para CSV da tabela"""
    # Preparar dados para CSV (convertendo emojis para texto)
    df_csv = df.copy()
    if 'Status' in df_csv.columns:
        df_csv['Status'] = df_csv['Status'].apply(formatar_status_csv)
    
    csv = df_csv.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📊 Download CSV",
        data=csv,
        file_name=f"{nome_arquivo}.csv",
        mime="text/csv",
        key=f"csv_{nome_arquivo}_{hash(str(df.values.tolist()))}"
    )

def gerar_dados_exemplo():
    """Gera dados de exemplo para desenvolvimento"""
    # Criar DataFrame base
    pontos_venda = ['Loja A', 'Loja B', 'Loja C', 'Loja D', 'Loja E', 
                    'Loja F', 'Loja G', 'Loja H', 'Loja I', 'Loja J']
    
    dados_exemplo = []
    anos = [2024, 2025]
    meses = obter_ordem_meses()
    
    # Gerar dados para cada combinação de ano, mês e ponto de venda
    for ano in anos:
        for mes in meses:
            for ponto in pontos_venda:
                # Gerar valores aleatórios com alguma lógica
                base_venda = np.random.randint(10000, 50000)
                # Variação sazonal - vendas maiores no fim do ano
                fator_sazonal = 1.0 + (obter_indice_mes(mes) / 12) * 0.5
                # Crescimento anual
                fator_anual = 1.0 + (ano - 2024) * 0.15
                
                venda = base_venda * fator_sazonal * fator_anual
                # Metas um pouco acima das vendas em média
                meta = venda * np.random.uniform(0.9, 1.3)
                
                dados_exemplo.append({
                    'Ponto de Venda': ponto,
                    'Ano': ano,
                    'Mês': mes,
                    'Venda': round(venda, 2),
                    'Metas': round(meta, 2)
                })
    
    return pd.DataFrame(dados_exemplo)

# Função para carregar dados reais do Google Sheets
@st.cache_data(ttl=600)
def carregar_dados_google_sheets():
    """Carrega dados da planilha Google Sheets"""
    try:
        # Tentar usar arquivo de credenciais local
        if os.path.exists(CREDENTIALS_PATH):
            st.info(f"🔐 Usando arquivo de credenciais local: {CREDENTIALS_PATH}")
            scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
            creds = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=scopes)
        # Tentar usar secrets do Streamlit
        elif hasattr(st, 'secrets') and "gcp_service_account" in st.secrets:
            st.info("🔐 Usando credenciais do Streamlit Secrets")
            credentials_info = dict(st.secrets["gcp_service_account"])
            scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
            creds = Credentials.from_service_account_info(credentials_info, scopes=scopes)
        else:
            # Se não há credenciais, mostrar erro
            st.error("❌ Credenciais não encontradas")
            st.error("💡 **Configure as credenciais seguindo uma das opções:**")
            st.error(f"1. Crie um arquivo de credenciais em: {CREDENTIALS_PATH}")
            st.error("2. Configure secrets no Streamlit Cloud")
            return pd.DataFrame()
        
        # Conectar ao Google Sheets
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SHEET_ID)
        worksheet = spreadsheet.worksheet(SHEET_NAME)
        
        df = get_as_dataframe(worksheet)
        df.columns = df.columns.str.strip()
        
        # Normalizar nomes dos meses
        if 'Mês' in df.columns:
            df['Mês'] = df['Mês'].apply(normalizar_nome_mes)
        
        # Normalizar nomes dos pontos de venda (remover espaços extras)
        if 'Ponto de Venda' in df.columns:
            df['Ponto de Venda'] = df['Ponto de Venda'].astype(str).str.strip()
        
        # Converter colunas numéricas
        numeric_cols = ['Ano', 'Venda', 'Metas']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remover linhas com valores nulos nas colunas críticas
        df = df.dropna(subset=[col for col in numeric_cols if col in df.columns])
        
        # Remover linhas onde Ponto de Venda é nulo ou vazio
        if 'Ponto de Venda' in df.columns:
            df = df[df['Ponto de Venda'].notna()]
            df = df[df['Ponto de Venda'] != '']
            df = df[df['Ponto de Venda'] != 'nan']
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.error("💡 **Possíveis soluções:**")
        st.error(f"1. Verifique se o arquivo de credenciais existe em: {CREDENTIALS_PATH}")
        st.error("2. Certifique-se que a planilha está compartilhada com o email da conta de serviço")
        st.error("3. Verifique se o ID da planilha e nome da aba estão corretos")
        return pd.DataFrame()

def aplicar_filtro_pontos_venda(df, pontos_selecionados):
    """Aplica filtro de pontos de venda no DataFrame"""
    if not pontos_selecionados:
        return df
    return df[df['Ponto de Venda'].isin(pontos_selecionados)]

def processar_soma_periodo_completo(df, ano_inicial, mes_inicial, ano_final, mes_final, pontos_selecionados=None):
    """Processa soma de todos os meses entre período inicial e final"""
    try:
        # Aplicar filtro de pontos de venda
        if pontos_selecionados:
            df = aplicar_filtro_pontos_venda(df, pontos_selecionados)
            st.info(f"🔍 **Filtro aplicado:** {len(pontos_selecionados)} ponto(s) de venda selecionado(s)")
        
        ordem_meses = obter_ordem_meses()
        
        # Criar lista de períodos
        periodos = []
        
        if ano_inicial == ano_final:
            # Mesmo ano
            idx_inicial = obter_indice_mes(mes_inicial)
            idx_final = obter_indice_mes(mes_final)
            for i in range(idx_inicial, idx_final + 1):
                periodos.append((ano_inicial, ordem_meses[i]))
        else:
            # Anos diferentes
            # Meses do ano inicial
            idx_inicial = obter_indice_mes(mes_inicial)
            for i in range(idx_inicial, 12):
                periodos.append((ano_inicial, ordem_meses[i]))
            
            # Anos intermediários (todos os meses)
            for ano in range(ano_inicial + 1, ano_final):
                for mes in ordem_meses:
                    periodos.append((ano, mes))
            
            # Meses do ano final
            idx_final = obter_indice_mes(mes_final)
            for i in range(0, idx_final + 1):
                periodos.append((ano_final, ordem_meses[i]))
        
        # Debug: mostrar períodos que serão somados
        st.info(f"🔍 **Períodos incluídos na soma:** {len(periodos)} meses")
        with st.expander("📅 Ver todos os períodos incluídos"):
            periodos_str = [f"{mes}/{ano}" for ano, mes in periodos]
            st.write(", ".join(periodos_str))
        
        # Filtrar dados pelos períodos
        dados_filtrados = []
        dados_debug = []
        
        for ano, mes in periodos:
            filtro = df[(df['Ano'] == ano) & (df['Mês'].str.strip() == mes)]
            if not filtro.empty:
                dados_filtrados.append(filtro)
                # Debug por período
                dados_debug.append({
                    'Período': f"{mes}/{ano}",
                    'Registros': len(filtro),
                    'Pontos': filtro['Ponto de Venda'].unique().tolist(),
                    'Total_Vendas': filtro['Venda'].sum(),
                    'Total_Metas': filtro['Metas'].sum()
                })
        
        # Debug: mostrar dados por período
        if dados_debug:
            with st.expander("🔍 Debug - Dados por período"):
                debug_df = pd.DataFrame(dados_debug)
                st.dataframe(debug_df, use_container_width=True)
        
        if not dados_filtrados:
            st.warning("⚠️ Nenhum dado encontrado para os períodos selecionados.")
            return pd.DataFrame()
        
        # Concatenar todos os dados
        df_completo = pd.concat(dados_filtrados, ignore_index=True)
        
        # Agregar por ponto de venda
        resumo = df_completo.groupby('Ponto de Venda').agg({
            'Venda': 'sum',
            'Metas': 'sum'
        }).reset_index()
        
        # Debug: mostrar resultado da agregação
        with st.expander("🔍 Debug - Resultado da agregação"):
            st.dataframe(resumo, use_container_width=True)
        
        # Calcular métricas
        resumo['Alcance de Meta (%)'] = (resumo['Venda'] / resumo['Metas'] * 100).round(2)
        resumo['Meta Batida'] = resumo['Alcance de Meta (%)'].apply(lambda x: '✅' if x >= 100 else '❌')
        
        # Formatação
        resumo['Venda_Formatada'] = resumo['Venda'].apply(formatar_real)
        resumo['Metas_Formatada'] = resumo['Metas'].apply(formatar_real)
        resumo['Alcance_Formatado'] = resumo['Alcance de Meta (%)'].apply(formatar_porcentagem)
        
        return resumo
        
    except Exception as e:
        st.error(f"Erro no processamento do período completo: {str(e)}")
        return pd.DataFrame()

def processar_periodo_vendas(df, ano=None, mes=None, pontos_selecionados=None):
    """Processa dados de vendas para um período específico"""
    try:
        # Aplicar filtro de pontos de venda
        if pontos_selecionados:
            df = aplicar_filtro_pontos_venda(df, pontos_selecionados)
        
        filtro = df.copy()
        if ano is not None:
            filtro = filtro[filtro['Ano'] == ano]
            if mes is not None:
                filtro = filtro[filtro['Mês'].str.strip() == mes]
        
        # Agrupa por ponto de venda
        resumo = filtro.groupby('Ponto de Venda').agg({
            'Venda': 'sum',
            'Metas': 'sum'
        }).reset_index()
        
        resumo['Alcance de Meta (%)'] = (resumo['Venda'] / resumo['Metas'] * 100).round(2)
        resumo['Meta Batida'] = resumo['Alcance de Meta (%)'].apply(lambda x: '✅' if x >= 100 else '❌')
        
        # Formatação
        resumo['Venda_Formatada'] = resumo['Venda'].apply(formatar_real)
        resumo['Metas_Formatada'] = resumo['Metas'].apply(formatar_real)
        resumo['Alcance_Formatado'] = resumo['Alcance de Meta (%)'].apply(formatar_porcentagem)
        
        return resumo
        
    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
        return pd.DataFrame()

def processar_periodo_metas(df, ano=None, mes=None, pontos_selecionados=None):
    """Processa dados de metas para um período específico"""
    try:
        # Aplicar filtro de pontos de venda
        if pontos_selecionados:
            df = aplicar_filtro_pontos_venda(df, pontos_selecionados)
        
        filtro = df.copy()
        if ano is not None:
            filtro = filtro[filtro['Ano'] == ano]
            if mes is not None:
                filtro = filtro[filtro['Mês'].str.strip() == mes]
                
        resumo = filtro.groupby('Ponto de Venda').agg({
            'Metas': 'sum'
        }).reset_index()
        
        # Formatação em Real
        resumo['Metas_Formatada'] = resumo['Metas'].apply(formatar_real)
        
        return resumo
        
    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
        return pd.DataFrame()

def criar_comparativo_vendas(df1, df2):
    """Cria dataframe comparativo de vendas entre dois períodos"""
    try:
        comparativo = pd.merge(
            df1,
            df2,
            on='Ponto de Venda',
            suffixes=('_p1', '_p2'),
            how='outer'
        ).fillna(0)
        
        comparativo['Variação Vendas (%)'] = ((comparativo['Venda_p2'] / comparativo['Venda_p1'] - 1) * 100).round(2)
        comparativo['Variação_Formatada'] = comparativo['Variação Vendas (%)'].apply(formatar_porcentagem)
        
        return comparativo
        
    except Exception as e:
        st.error(f"Erro na comparação: {str(e)}")
        return pd.DataFrame()

def criar_comparativo_metas(df1, df2):
    """Cria dataframe comparativo de metas entre dois períodos"""
    try:
        comparativo = pd.merge(
            df1,
            df2,
            on='Ponto de Venda',
            suffixes=('_p1', '_p2'),
            how='outer'
        ).fillna(0)
        
        comparativo['Evolução Metas (%)'] = ((comparativo['Metas_p2'] / comparativo['Metas_p1'] - 1) * 100).round(2)
        comparativo['Crescimento'] = comparativo['Evolução Metas (%)'].apply(
            lambda x: '📈 Cresceu' if x > 0 else '📉 Diminuiu' if x < 0 else '➡️ Manteve'
        )
        
        return comparativo
        
    except Exception as e:
        st.error(f"Erro na comparação: {str(e)}")
        return pd.DataFrame()

def criar_grafico_vendas_comparacao(df, periodo1_label, periodo2_label):
    """Cria gráfico de barras comparativo de vendas - modo comparação"""
    try:
        fig = go.Figure()
        
        pontos = df['Ponto de Venda'].tolist()
        
        # Barras
        fig.add_trace(go.Bar(
            x=pontos,
            y=df['Venda_p1'],
            name=f'{periodo1_label}',
            marker_color='#1f77b4',
            text=[formatar_real(v) for v in df['Venda_p1']],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            x=pontos,
            y=df['Venda_p2'],
            name=f'{periodo2_label}',
            marker_color='#ff7f0e',
            text=[formatar_real(v) for v in df['Venda_p2']],
            textposition='outside'
        ))
        
        # Linha de evolução
        fig.add_trace(go.Scatter(
            x=pontos,
            y=df['Venda_p2'],
            mode='lines+markers',
            name='Linha de Evolução',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Comparação de Vendas entre Períodos',
            barmode='group',
            height=500,
            xaxis_title='Pontos de Venda',
            yaxis_title='Valor (R$)',
            yaxis2=dict(
                title='Evolução',
                overlaying='y',
                side='right'
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro na criação do gráfico: {str(e)}")
        return go.Figure()

def criar_grafico_vendas_soma(df_soma, periodo_label):
    """Cria gráfico de barras para modo soma"""
    try:
        fig = go.Figure()
        
        pontos = df_soma['Ponto de Venda'].tolist()
        
        # Barra da soma total
        fig.add_trace(go.Bar(
            x=pontos,
            y=df_soma['Venda'],
            name='Total Vendas',
            marker_color='#2ca02c',
            text=[formatar_real(v) for v in df_soma['Venda']],
            textposition='outside'
        ))
        
        # Linha das metas
        fig.add_trace(go.Scatter(
            x=pontos,
            y=df_soma['Metas'],
            mode='lines+markers',
            name='Metas',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Soma do Período: {periodo_label}',
            height=500,
            xaxis_title='Pontos de Venda',
            yaxis_title='Vendas (R$)',
            yaxis2=dict(
                title='Metas (R$)',
                overlaying='y',
                side='right'
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro na criação do gráfico: {str(e)}")
        return go.Figure()

def criar_grafico_crescimento(df1, df2, periodo1_label, periodo2_label):
    """Cria gráfico de crescimento/decréscimo das somas"""
    try:
        total_p1 = df1['Venda'].sum()
        total_p2 = df2['Venda'].sum()
        variacao = ((total_p2 / total_p1 - 1) * 100) if total_p1 > 0 else 0
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[periodo1_label, periodo2_label],
            y=[total_p1, total_p2],
            marker_color=['#1f77b4', '#ff7f0e'],
            text=[formatar_real(total_p1), formatar_real(total_p2)],
            textposition='outside'
        ))
        
        # Linha de evolução
        fig.add_trace(go.Scatter(
            x=[periodo1_label, periodo2_label],
            y=[total_p1, total_p2],
            mode='lines+markers',
            name=f'Evolução: {formatar_porcentagem(variacao)}',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title=f'Evolução Total de Vendas - {formatar_porcentagem(variacao)}',
            height=400,
            xaxis_title='Períodos',
            yaxis_title='Valor Total (R$)'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro na criação do gráfico: {str(e)}")
        return go.Figure()

def criar_grafico_metas(df, periodo1_label, periodo2_label):
    """Cria gráfico de barras comparativo de metas"""
    try:
        fig = go.Figure()
        
        pontos = df['Ponto de Venda'].tolist()
        
        # Barras
        fig.add_trace(go.Bar(
            x=pontos,
            y=df['Metas_p1'],
            name=f'{periodo1_label}',
            marker_color='#2ca02c',
            text=[formatar_real(v) for v in df['Metas_p1']],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            x=pontos,
            y=df['Metas_p2'],
            name=f'{periodo2_label}',
            marker_color='#d62728',
            text=[formatar_real(v) for v in df['Metas_p2']],
            textposition='outside'
        ))
        
        # Linha de evolução
        fig.add_trace(go.Scatter(
            x=pontos,
            y=df['Metas_p2'],
            mode='lines+markers',
            name='Linha de Evolução das Metas',
            line=dict(color='purple', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Evolução das Metas entre Períodos',
            barmode='group',
            height=500,
            xaxis_title='Pontos de Venda',
            yaxis_title='Valor (R$)',
            yaxis2=dict(
                title='Evolução',
                overlaying='y',
                side='right'
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro na criação do gráfico: {str(e)}")
        return go.Figure()

def main():
    st.title("Dashboard Comercial")
    
    # Verificar status das dependências
    if PDF_TABELA_DISPONIVEL:
        st.success("✅ PDF de tabelas disponível")
    else:
        st.warning("⚠️ Para habilitar PDF de tabelas, execute: pip install reportlab")
    
    # IMPORTANTE: Movemos o checkbox para fora da função cacheada
    st.sidebar.header("🔧 Configurações")
    usar_dados_exemplo = st.sidebar.checkbox(
        "Usar dados de exemplo", 
        value=False,
        help="Ative para usar dados fictícios em vez de conectar ao Google Sheets"
    )
    
    # Carregar dados com base na escolha do usuário
    if usar_dados_exemplo:
        st.info("🔄 Usando dados de exemplo para desenvolvimento")
        df_base = gerar_dados_exemplo()
        st.success(f"✅ Dados de exemplo carregados! {len(df_base)} registros gerados.")
    else:
        df_base = carregar_dados_google_sheets()
        if not df_base.empty:
            st.success(f"✅ Dados carregados com sucesso! {len(df_base)} registros encontrados.")
    
    if df_base.empty:
        st.error("❌ Não foi possível carregar os dados.")
        return
    
    # Debug: mostrar estrutura dos dados
    with st.expander("🔍 Debug - Estrutura dos Dados"):
        st.write("**Colunas disponíveis:**", df_base.columns.tolist())
        st.write("**Total de registros:**", len(df_base))
        st.write("**Primeiras linhas:**")
        st.dataframe(df_base.head())
        if 'Mês' in df_base.columns:
            st.write("**Meses únicos:**", sorted(df_base['Mês'].unique()))
        if 'Ponto de Venda' in df_base.columns:
            st.write("**Pontos de Venda únicos:**", sorted(df_base['Ponto de Venda'].unique()))
            st.write("**Total de pontos:**", df_base['Ponto de Venda'].nunique())
    
    # Sidebar com filtros
    st.sidebar.header("🔧 Filtros")
    
    tipo_analise = st.sidebar.selectbox(
        "Tipo de Análise",
        ["Venda", "Meta"]
    )
    
    # Filtro de Pontos de Venda
    if 'Ponto de Venda' in df_base.columns:
        pontos_disponiveis = sorted(df_base['Ponto de Venda'].unique())
        pontos_selecionados = st.sidebar.multiselect(
            "🏪 Filtrar por Pontos de Venda",
            options=pontos_disponiveis,
            default=pontos_disponiveis,  # Todos selecionados por padrão
            help="Selecione um ou mais pontos de venda para análise"
        )
        
        if not pontos_selecionados:
            st.warning("⚠️ Selecione pelo menos um ponto de venda.")
            return
        
        # Mostrar filtro aplicado
        if len(pontos_selecionados) < len(pontos_disponiveis):
            st.info(f"🔍 **Filtro ativo:** {len(pontos_selecionados)} de {len(pontos_disponiveis)} pontos selecionados")
            with st.expander("Ver pontos selecionados"):
                st.write(", ".join(pontos_selecionados))
    else:
        pontos_selecionados = None
    
    with st.container():
        
        col1_p1_month_year, col2_p2_month_year = st.columns(2)
        
        # Período 1
        with col1_p1_month_year:
            if 'Ano' in df_base.columns:
                ano_p1 = int(st.selectbox(
                    "Ano (Período 1)", 
                    options=sorted(df_base["Ano"].unique()),
                    key="ano_p1"
                ))
                meses_disponiveis = sorted(df_base[df_base["Ano"] == ano_p1]["Mês"].str.strip().unique(), 
                                         key=lambda x: obter_indice_mes(x))
                mes_p1_selected = str(st.selectbox(
                    f"Mês ({ano_p1})", 
                    options=meses_disponiveis,
                    key="mes_p1"
                ))
            else:
                st.error("Coluna 'Ano' não encontrada nos dados")
                return
            
        # Período 2    
        with col2_p2_month_year:
            anos_validos = [a for a in sorted(df_base["Ano"].unique()) if a >= ano_p1]
            
            ano_p2 = int(st.selectbox(
                "Ano (Período 2)",
                options=anos_validos,
                index=(len(anos_validos)-1 if len(anos_validos) >= 1 else 0),
                key="ano_p2"
            ))
                
            meses_finais = sorted(df_base[df_base["Ano"] == ano_p2]["Mês"].str.strip().unique(),
                                key=lambda x: obter_indice_mes(x))
            mes_p2_selected = str(st.selectbox(
                f"Mês ({ano_p2})", 
                options=meses_finais,
                key="mes_p2"
            ))
        
        # Labels dos períodos
        periodo1_label = f"{mes_p1_selected}/{ano_p1}"
        periodo2_label = f"{mes_p2_selected}/{ano_p2}"
        
        if tipo_analise == "Venda":
            # Checkbox para modo de análise
            modo_soma = st.checkbox("🔄 Modo Soma (somar todos os meses do período inicial até o final)", value=False)
            
            if modo_soma:
                # NOVA IMPLEMENTAÇÃO: Checkbox para escolher entre análise de 1 ou 2 períodos de soma
                comparar_dois_periodos_soma = st.checkbox("📊 Comparar dois períodos de soma", value=False)
                
                if comparar_dois_periodos_soma:
                    # Modo de comparação entre dois períodos de soma
                    st.info("📈 **Modo Soma Comparativo**: Comparando dois períodos de soma")
                    
                    # Definir os períodos de soma
                    col1_soma, col2_soma = st.columns(2)
                    
                    with col1_soma:
                        st.subheader("Primeiro Período de Soma")
                        # Seleção do período inicial da primeira soma
                        col_inicio_p1, col_fim_p1 = st.columns(2)
                        with col_inicio_p1:
                            ano_inicio_p1 = int(st.selectbox(
                                "Ano Inicial", 
                                options=sorted(df_base["Ano"].unique()),
                                key="ano_inicio_p1"
                            ))
                            meses_inicio_p1 = sorted(df_base[df_base["Ano"] == ano_inicio_p1]["Mês"].str.strip().unique(), 
                                                 key=lambda x: obter_indice_mes(x))
                            mes_inicio_p1 = str(st.selectbox(
                                "Mês Inicial", 
                                options=meses_inicio_p1,
                                key="mes_inicio_p1"
                            ))
                        
                        with col_fim_p1:
                            anos_fim_p1 = [a for a in sorted(df_base["Ano"].unique()) if a >= ano_inicio_p1]
                            ano_fim_p1 = int(st.selectbox(
                                "Ano Final", 
                                options=anos_fim_p1,
                                key="ano_fim_p1"
                            ))
                            
                            # Filtrar meses disponíveis para o ano final
                            meses_fim_p1 = sorted(df_base[df_base["Ano"] == ano_fim_p1]["Mês"].str.strip().unique(),
                                               key=lambda x: obter_indice_mes(x))
                            
                            # Se mesmo ano, filtrar meses após o mês inicial
                            if ano_fim_p1 == ano_inicio_p1:
                                idx_mes_inicio = obter_indice_mes(mes_inicio_p1)
                                meses_fim_p1 = [m for m in meses_fim_p1 if obter_indice_mes(m) >= idx_mes_inicio]
                            
                            mes_fim_p1 = str(st.selectbox(
                                "Mês Final", 
                                options=meses_fim_p1,
                                key="mes_fim_p1"
                            ))
                    
                    with col2_soma:
                        st.subheader("Segundo Período de Soma")
                        # Seleção do período inicial da segunda soma
                        col_inicio_p2, col_fim_p2 = st.columns(2)
                        with col_inicio_p2:
                            ano_inicio_p2 = int(st.selectbox(
                                "Ano Inicial", 
                                options=sorted(df_base["Ano"].unique()),
                                key="ano_inicio_p2"
                            ))
                            meses_inicio_p2 = sorted(df_base[df_base["Ano"] == ano_inicio_p2]["Mês"].str.strip().unique(), 
                                                 key=lambda x: obter_indice_mes(x))
                            mes_inicio_p2 = str(st.selectbox(
                                "Mês Inicial", 
                                options=meses_inicio_p2,
                                key="mes_inicio_p2"
                            ))
                        
                        with col_fim_p2:
                            anos_fim_p2 = [a for a in sorted(df_base["Ano"].unique()) if a >= ano_inicio_p2]
                            ano_fim_p2 = int(st.selectbox(
                                "Ano Final", 
                                options=anos_fim_p2,
                                key="ano_fim_p2"
                            ))
                            
                            # Filtrar meses disponíveis para o ano final
                            meses_fim_p2 = sorted(df_base[df_base["Ano"] == ano_fim_p2]["Mês"].str.strip().unique(),
                                               key=lambda x: obter_indice_mes(x))
                            
                            # Se mesmo ano, filtrar meses após o mês inicial
                            if ano_fim_p2 == ano_inicio_p2:
                                idx_mes_inicio = obter_indice_mes(mes_inicio_p2)
                                meses_fim_p2 = [m for m in meses_fim_p2 if obter_indice_mes(m) >= idx_mes_inicio]
                            
                            mes_fim_p2 = str(st.selectbox(
                                "Mês Final", 
                                options=meses_fim_p2,
                                key="mes_fim_p2"
                            ))
                    
                    # Processamento dos dois períodos de soma
                    df_soma_p1 = processar_soma_periodo_completo(df_base, ano_inicio_p1, mes_inicio_p1, ano_fim_p1, mes_fim_p1, pontos_selecionados)
                    df_soma_p2 = processar_soma_periodo_completo(df_base, ano_inicio_p2, mes_inicio_p2, ano_fim_p2, mes_fim_p2, pontos_selecionados)
                    
                    # Labels para os períodos de soma
                    periodo_soma_p1_label = f"{mes_inicio_p1}/{ano_inicio_p1} até {mes_fim_p1}/{ano_fim_p1}"
                    periodo_soma_p2_label = f"{mes_inicio_p2}/{ano_inicio_p2} até {mes_fim_p2}/{ano_fim_p2}"
                    
                    if not df_soma_p1.empty and not df_soma_p2.empty:
                        # Exibir resultados dos dois períodos de soma
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(f"🔄 Soma do Período 1 ({periodo_soma_p1_label})")
                            df_soma_p1_display = df_soma_p1[['Ponto de Venda', 'Venda_Formatada', 'Metas_Formatada', 'Alcance_Formatado', 'Meta Batida']].copy()
                            df_soma_p1_display.columns = ['Ponto de Venda', 'Vendas Totais', 'Metas Totais', 'Alcance', 'Status']
                            st.dataframe(df_soma_p1_display, use_container_width=True)
                            
                            col_pdf, col_csv = st.columns(2)
                            with col_pdf:
                                botao_download_pdf_tabela(df_soma_p1_display, f"Soma - {periodo_soma_p1_label}", f"soma_p1_{ano_inicio_p1}{mes_inicio_p1}_{ano_fim_p1}{mes_fim_p1}")
                            with col_csv:
                                botao_download_csv_tabela(df_soma_p1_display, f"soma_p1_{ano_inicio_p1}{mes_inicio_p1}_{ano_fim_p1}{mes_fim_p1}")
                            
                            # Totais do período 1
                            total_vendas_p1 = df_soma_p1['Venda'].sum()
                            total_metas_p1 = df_soma_p1['Metas'].sum()
                            alcance_p1 = (total_vendas_p1 / total_metas_p1 * 100) if total_metas_p1 > 0 else 0
                            st.metric("Total Vendas", formatar_real(total_vendas_p1))
                            st.metric("Total Metas", formatar_real(total_metas_p1))
                            st.metric("Alcance Geral", formatar_porcentagem(alcance_p1))
                        
                        with col2:
                            st.subheader(f"🔄 Soma do Período 2 ({periodo_soma_p2_label})")
                            df_soma_p2_display = df_soma_p2[['Ponto de Venda', 'Venda_Formatada', 'Metas_Formatada', 'Alcance_Formatado', 'Meta Batida']].copy()
                            df_soma_p2_display.columns = ['Ponto de Venda', 'Vendas Totais', 'Metas Totais', 'Alcance', 'Status']
                            st.dataframe(df_soma_p2_display, use_container_width=True)
                            
                            col_pdf, col_csv = st.columns(2)
                            with col_pdf:
                                botao_download_pdf_tabela(df_soma_p2_display, f"Soma - {periodo_soma_p2_label}", f"soma_p2_{ano_inicio_p2}{mes_inicio_p2}_{ano_fim_p2}{mes_fim_p2}")
                            with col_csv:
                                botao_download_csv_tabela(df_soma_p2_display, f"soma_p2_{ano_inicio_p2}{mes_inicio_p2}_{ano_fim_p2}{mes_fim_p2}")
                            
                            # Totais do período 2
                            total_vendas_p2 = df_soma_p2['Venda'].sum()
                            total_metas_p2 = df_soma_p2['Metas'].sum()
                            alcance_p2 = (total_vendas_p2 / total_metas_p2 * 100) if total_metas_p2 > 0 else 0
                            evolucao = ((total_vendas_p2 / total_vendas_p1 - 1) * 100) if total_vendas_p1 > 0 else 0
                            st.metric("Total Vendas", formatar_real(total_vendas_p2))
                            st.metric("Total Metas", formatar_real(total_metas_p2))
                            st.metric("Alcance Geral", formatar_porcentagem(alcance_p2))
                            st.metric("Evolução", formatar_porcentagem(evolucao), 
                                     delta="📈 Cresceu" if evolucao > 0 else "📉 Diminuiu" if evolucao < 0 else "➡️ Manteve")
                        
                        # Criar comparativo entre os dois períodos de soma
                        comparativo_soma = criar_comparativo_vendas(df_soma_p1, df_soma_p2)
                        
                        st.subheader("📈 Comparativo entre Períodos de Soma")
                        col_comp, col_botoes_comp = st.columns([3, 1])
                        with col_comp:
                            df_comp_display = comparativo_soma[['Ponto de Venda', 'Venda_p1', 'Venda_p2', 'Variação_Formatada']].copy()
                            df_comp_display['Venda_p1'] = df_comp_display['Venda_p1'].apply(formatar_real)
                            df_comp_display['Venda_p2'] = df_comp_display['Venda_p2'].apply(formatar_real)
                            df_comp_display.columns = ['Ponto de Venda', f'Soma {periodo_soma_p1_label}', f'Soma {periodo_soma_p2_label}', 'Variação']
                            st.dataframe(df_comp_display, use_container_width=True)
                        with col_botoes_comp:
                            botao_download_pdf_tabela(df_comp_display, f"Comparativo Somas - {periodo_soma_p1_label} vs {periodo_soma_p2_label}", 
                                                    f"comp_somas_{ano_inicio_p1}{mes_inicio_p1}_{ano_fim_p2}{mes_fim_p2}")
                            botao_download_csv_tabela(df_comp_display, f"comp_somas_{ano_inicio_p1}{mes_inicio_p1}_{ano_fim_p2}{mes_fim_p2}")
                        
                        # Gráficos comparativos
                        fig_comparacao = criar_grafico_vendas_comparacao(comparativo_soma, 
                                                                        f"Soma {periodo_soma_p1_label}", 
                                                                        f"Soma {periodo_soma_p2_label}")
                        st.plotly_chart(fig_comparacao, use_container_width=True)
                        
                        # Gráfico de crescimento
                        fig_crescimento = criar_grafico_crescimento(df_soma_p1, df_soma_p2, 
                                                                  f"Soma {periodo_soma_p1_label}", 
                                                                  f"Soma {periodo_soma_p2_label}")
                        st.plotly_chart(fig_crescimento, use_container_width=True)
                    else:
                        st.warning("⚠️ Nenhum dado encontrado para um ou ambos os períodos selecionados.")
                
                else:
                    # Modo soma original (um único período)
                    st.info(f"📊 **Modo Soma**: Somando todos os meses de {periodo1_label} até {periodo2_label} por ponto de venda")
                    
                    # Processamento do período completo
                    df_soma_completa = processar_soma_periodo_completo(df_base, ano_p1, mes_p1_selected, ano_p2, mes_p2_selected, pontos_selecionados)
                    
                    if not df_soma_completa.empty:
                        st.subheader(f"🔄 Soma Completa do Período ({periodo1_label} até {periodo2_label})")
                        
                        # Tabela da soma completa
                        col_tabela, col_botoes = st.columns([3, 1])
                        with col_tabela:
                            df_soma_display = df_soma_completa[['Ponto de Venda', 'Venda_Formatada', 'Metas_Formatada', 'Alcance_Formatado', 'Meta Batida']].copy()
                            df_soma_display.columns = ['Ponto de Venda', 'Vendas Totais', 'Metas Totais', 'Alcance', 'Status']
                            st.dataframe(df_soma_display, use_container_width=True)
                        
                        with col_botoes:
                            botao_download_pdf_tabela(df_soma_display, f"Soma Completa - {periodo1_label} até {periodo2_label}", f"soma_completa_{ano_p1}{mes_p1_selected}_{ano_p2}{mes_p2_selected}")
                            botao_download_csv_tabela(df_soma_display, f"soma_completa_{ano_p1}{mes_p1_selected}_{ano_p2}{mes_p2_selected}")
                        
                        # Totais da soma completa
                        total_vendas_soma = df_soma_completa['Venda'].sum()
                        total_metas_soma = df_soma_completa['Metas'].sum()
                        alcance_soma = (total_vendas_soma / total_metas_soma * 100) if total_metas_soma > 0 else 0
                        
                        col1_total, col2_total, col3_total = st.columns(3)
                        with col1_total:
                            st.metric("Total Vendas", formatar_real(total_vendas_soma))
                        with col2_total:
                            st.metric("Total Metas", formatar_real(total_metas_soma))
                        with col3_total:
                            st.metric("Alcance Geral", formatar_porcentagem(alcance_soma))
                        
                        # Gráfico modo soma completa
                        periodo_completo_label = f"{periodo1_label} até {periodo2_label}"
                        fig_soma_completa = criar_grafico_vendas_soma(df_soma_completa, periodo_completo_label)
                        st.plotly_chart(fig_soma_completa, use_container_width=True)
                    else:
                        st.warning("⚠️ Nenhum dado encontrado para o período selecionado.")
            
            else:
                # Modo comparação original (sem alterações)
                st.info("📈 **Modo Comparação**: Comparando período 1 vs período 2 individualmente")
                
                # Processamento dos dados de vendas
                periodo01 = processar_periodo_vendas(df_base, ano=ano_p1, mes=mes_p1_selected, pontos_selecionados=pontos_selecionados)
                periodo02 = processar_periodo_vendas(df_base, ano=ano_p2, mes=mes_p2_selected, pontos_selecionados=pontos_selecionados)
                
                # Exibição dos resultados
                if not periodo01.empty and not periodo02.empty:
                    comparativo = criar_comparativo_vendas(periodo01, periodo02)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"📊 {periodo1_label}")
                        df_display = periodo01[['Ponto de Venda', 'Venda_Formatada', 'Metas_Formatada', 'Alcance_Formatado', 'Meta Batida']].copy()
                        df_display.columns = ['Ponto de Venda', 'Vendas', 'Metas', 'Alcance', 'Status']
                        st.dataframe(df_display, use_container_width=True)
                        
                        col_pdf, col_csv = st.columns(2)
                        with col_pdf:
                            botao_download_pdf_tabela(df_display, f"Vendas - {periodo1_label}", f"vendas_{ano_p1}{mes_p1_selected}")
                        with col_csv:
                            botao_download_csv_tabela(df_display, f"vendas_{ano_p1}{mes_p1_selected}")
                        
                        # Totais P1
                        total_vendas_p1 = periodo01['Venda'].sum()
                        total_metas_p1 = periodo01['Metas'].sum()
                        alcance_p1 = (total_vendas_p1 / total_metas_p1 * 100) if total_metas_p1 > 0 else 0
                        st.metric("Total Vendas", formatar_real(total_vendas_p1))
                        st.metric("Total Metas", formatar_real(total_metas_p1))
                        st.metric("Alcance Geral", formatar_porcentagem(alcance_p1))
                    
                    with col2:
                        st.subheader(f"📊 {periodo2_label}")
                        df_display = periodo02[['Ponto de Venda', 'Venda_Formatada', 'Metas_Formatada', 'Alcance_Formatado', 'Meta Batida']].copy()
                        df_display.columns = ['Ponto de Venda', 'Vendas', 'Metas', 'Alcance', 'Status']
                        st.dataframe(df_display, use_container_width=True)
                        
                        col_pdf, col_csv = st.columns(2)
                        with col_pdf:
                            botao_download_pdf_tabela(df_display, f"Vendas - {periodo2_label}", f"vendas_{ano_p2}{mes_p2_selected}")
                        with col_csv:
                            botao_download_csv_tabela(df_display, f"vendas_{ano_p2}{mes_p2_selected}")
                        
                        # Totais P2
                        total_vendas_p2 = periodo02['Venda'].sum()
                        total_metas_p2 = periodo02['Metas'].sum()
                        alcance_p2 = (total_vendas_p2 / total_metas_p2 * 100) if total_metas_p2 > 0 else 0
                        evolucao = ((total_vendas_p2 / total_vendas_p1 - 1) * 100) if total_vendas_p1 > 0 else 0
                        st.metric("Total Vendas", formatar_real(total_vendas_p2))
                        st.metric("Total Metas", formatar_real(total_metas_p2))
                        st.metric("Alcance Geral", formatar_porcentagem(alcance_p2))
                        st.metric("Evolução", formatar_porcentagem(evolucao), 
                                 delta="📈 Cresceu" if evolucao > 0 else "📉 Diminuiu" if evolucao < 0 else "➡️ Manteve")
                    
                    st.subheader("📈 Comparativo de Vendas")
                    col_comp, col_botoes_comp = st.columns([3, 1])
                    with col_comp:
                        df_comp_display = comparativo[['Ponto de Venda', 'Venda_p1', 'Venda_p2', 'Variação_Formatada']].copy()
                        # Correção: Removida linha duplicada de formatação
                        df_comp_display['Venda_p1'] = df_comp_display['Venda_p1'].apply(formatar_real)
                        df_comp_display['Venda_p2'] = df_comp_display['Venda_p2'].apply(formatar_real)
                        df_comp_display.columns = ['Ponto de Venda', periodo1_label, periodo2_label, 'Variação']
                        st.dataframe(df_comp_display, use_container_width=True)
                    with col_botoes_comp:
                        botao_download_pdf_tabela(df_comp_display, f"Comparativo - {periodo1_label} vs {periodo2_label}", f"comparativo_{ano_p1}{mes_p1_selected}_{ano_p2}{mes_p2_selected}")
                        botao_download_csv_tabela(df_comp_display, f"comparativo_{ano_p1}{mes_p1_selected}_{ano_p2}{mes_p2_selected}")
                    
                    # Gráficos modo comparação
                    fig_comparacao = criar_grafico_vendas_comparacao(comparativo, periodo1_label, periodo2_label)
                    st.plotly_chart(fig_comparacao, use_container_width=True)
                    
                    # Gráfico de crescimento
                    fig_crescimento = criar_grafico_crescimento(periodo01, periodo02, periodo1_label, periodo2_label)
                    st.plotly_chart(fig_crescimento, use_container_width=True)
                else:
                    st.warning("⚠️ Nenhum dado encontrado para os períodos selecionados.")
        
        elif tipo_analise == "Meta":
            # Processamento dos dados de metas
            periodo01 = processar_periodo_metas(df_base, ano=ano_p1, mes=mes_p1_selected, pontos_selecionados=pontos_selecionados)
            periodo02 = processar_periodo_metas(df_base, ano=ano_p2, mes=mes_p2_selected, pontos_selecionados=pontos_selecionados)
            
            # Exibição dos resultados
            if not periodo01.empty and not periodo02.empty:
                comparativo = criar_comparativo_metas(periodo01, periodo02)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"🎯 {periodo1_label}")
                    df_display = periodo01[['Ponto de Venda', 'Metas_Formatada']].copy()
                    df_display.columns = ['Ponto de Venda', 'Metas']
                    st.dataframe(df_display, use_container_width=True)
                    
                    col_pdf, col_csv = st.columns(2)
                    with col_pdf:
                        botao_download_pdf_tabela(df_display, f"Metas - {periodo1_label}", f"metas_{ano_p1}{mes_p1_selected}")
                    with col_csv:
                        botao_download_csv_tabela(df_display, f"metas_{ano_p1}{mes_p1_selected}")
                
                with col2:
                    st.subheader(f"🎯 {periodo2_label}")
                    df_display = periodo02[['Ponto de Venda', 'Metas_Formatada']].copy()
                    df_display.columns = ['Ponto de Venda', 'Metas']
                    st.dataframe(df_display, use_container_width=True)
                    
                    col_pdf, col_csv = st.columns(2)
                    with col_pdf:
                        botao_download_pdf_tabela(df_display, f"Metas - {periodo2_label}", f"metas_{ano_p2}{mes_p2_selected}")
                    with col_csv:
                        botao_download_csv_tabela(df_display, f"metas_{ano_p2}{mes_p2_selected}")
                
                st.subheader("📊 Evolução das Metas")
                col_evo_meta, col_botoes_evo_meta = st.columns([3, 1])
                with col_evo_meta:
                    df_comp_display = comparativo[['Ponto de Venda', 'Metas_p1', 'Metas_p2', 'Evolução Metas (%)', 'Crescimento']].copy()
                    df_comp_display['Metas_p1'] = df_comp_display['Metas_p1'].apply(formatar_real)
                    df_comp_display['Metas_p2'] = df_comp_display['Metas_p2'].apply(formatar_real)
                    df_comp_display.columns = ['Ponto de Venda', periodo1_label, periodo2_label, 'Evolução (%)', 'Tendência']
                    st.dataframe(df_comp_display, use_container_width=True)
                
                with col_botoes_evo_meta:
                    botao_download_pdf_tabela(df_comp_display, f"Evolução Metas - {periodo1_label} vs {periodo2_label}", f"evolucao_metas_{ano_p1}{mes_p1_selected}_{ano_p2}{mes_p2_selected}")
                    botao_download_csv_tabela(df_comp_display, f"evolucao_metas_{ano_p1}{mes_p1_selected}_{ano_p2}{mes_p2_selected}")

                if not comparativo.empty:
                    fig = criar_grafico_metas(comparativo, periodo1_label, periodo2_label)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Nenhum dado encontrado para os períodos selecionados.")
            else:
                st.warning("⚠️ Nenhum dado encontrado para os períodos selecionados.")

if __name__ == "__main__":
    main()

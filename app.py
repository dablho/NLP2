import streamlit as st
import pandas as pd
import psycopg2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import unicodedata
from collections import Counter, defaultdict
import os
import networkx as nx
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Definir o diretório para armazenar os dados do NLTK
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Adicionar o diretório ao caminho de dados do NLTK
nltk.data.path.append(nltk_data_dir)
nltk.download('all', download_dir=nltk_data_dir)

# Configuração da página
st.set_page_config(page_title="Análise de Sentimentos", layout="centered")


# Download dos recursos necessários do NLTK
@st.cache_resource
def setup_sentiment_model():
    # Usando BERTimbau ou outro modelo pré-treinado para português
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "neuralmind/bert-base-portuguese-cased", num_labels=3
    )

    # Cria o pipeline de classificação
    classifier = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True
    )

    return classifier


def setup_nltk():
    nltk.download("vader_lexicon")
    nltk.download("punkt")
    nltk.download("stopwords")
    return SentimentIntensityAnalyzer()


# Conexão com o banco de dados
def get_data_from_db():
    try:
        conn = psycopg2.connect(
             host=os.getenv("host"),
             database=os.getenv("database"),
             user=os.getenv("user"),
             password=os.getenv("password")
        )
        query = "SELECT * FROM prova.tabela_tcc"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None


def generate_wordcloud(texts):
    # Combina todos os textos em um único string
    text = " ".join(texts)

    # Cria a wordcloud sem pré-processamento
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", max_words=200
    ).generate(text)

    return wordcloud


def get_word_frequency(texts, top_n=10):
    # Junta todos os textos
    all_words = " ".join(texts).split()

    # Conta a frequência das palavras
    word_freq = Counter(all_words)

    # Pega as top N palavras mais frequentes
    top_words = dict(
        sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    return top_words


def get_processed_word_frequency(texts, top_n=10):
    # Pré-processa e tokeniza todos os textos
    processed_words = []
    for text in texts:
        processed_text = preprocess_text_for_wordcloud(text)
        processed_words.extend(processed_text.split())

    # Conta a frequência das palavras
    word_freq = Counter(processed_words)

    # Pega as top N palavras mais frequentes
    top_words = dict(
        sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    return top_words


# Pré-processamento de texto
def preprocess_text(text):
    # Remove acentos
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    # Converte para minúsculas
    text = text.lower()
    # Remove pontuação
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def preprocess_text_for_wordcloud(text):
    # Remove acentos
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    # Converte para minúsculas
    text = text.lower()

    # Remove pontuação
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenização
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("portuguese"))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


def generate_processed_wordcloud(texts):
    # Pré-processa todos os textos
    processed_texts = [preprocess_text_for_wordcloud(text) for text in texts]

    # Combina todos os textos processados
    text = " ".join(processed_texts)

    # Cria a wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        colormap="viridis",
        prefer_horizontal=0.7,
    ).generate(text)

    return wordcloud


##############


def get_tetragrams(texts, top_n=10):
    # Pré-processa e obtém os tetragramas
    tetragrams = []
    for text in texts:
        # Pré-processamento
        processed_text = preprocess_text_with_custom_stopwords(text)
        words = processed_text.split()

        # Gera tetragramas
        for i in range(len(words) - 3):
            tetragrams.append(tuple(words[i : i + 4]))

    # Conta frequência
    tetragrams_freq = Counter(tetragrams)

    # Retorna os top N mais frequentes
    return dict(
        sorted(tetragrams_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )


def create_tetragrams_graph(texts):
    # Obtém os tetragramas mais frequentes
    top_tetragrams = get_tetragrams(texts)

    # Cria o grafo
    G = nx.DiGraph()

    # Adiciona nós e arestas
    for tetagram, weight in top_tetragrams.items():
        for i in range(3):
            G.add_edge(tetagram[i], tetagram[i + 1], weight=weight)

    return G, top_tetragrams


###############


def preprocess_text_with_custom_stopwords(text):
    # Stopwords personalizadas
    custom_stopwords = {"ja", "so", "pra"}

    # Pega as stopwords padrão do português
    stop_words = set(stopwords.words("portuguese"))

    # Adiciona as stopwords personalizadas
    stop_words.update(custom_stopwords)

    # Remove acentos
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    # Converte para minúsculas
    text = text.lower()

    # Remove pontuação
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenização
    tokens = word_tokenize(text)

    # Remove stopwords (incluindo as personalizadas)
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


def generate_custom_wordcloud(texts):
    # Pré-processa todos os textos
    processed_texts = [preprocess_text_with_custom_stopwords(text) for text in texts]

    # Combina todos os textos processados
    text = " ".join(processed_texts)

    # Cria a wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        colormap="viridis",
        prefer_horizontal=0.7,
    ).generate(text)

    return wordcloud


def get_custom_word_frequency(texts, top_n=10):
    # Pré-processa e tokeniza todos os textos
    processed_words = []
    for text in texts:
        processed_text = preprocess_text_with_custom_stopwords(text)
        processed_words.extend(processed_text.split())

    # Conta a frequência das palavras
    word_freq = Counter(processed_words)

    # Pega as top N palavras mais frequentes
    top_words = dict(
        sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    return top_words


# Análise de sentimento
def analyze_sentiment(text, sia):
    # Pré-processa o texto
    processed_text = preprocess_text(text)

    # Calcula os scores de sentimento
    scores = sia.polarity_scores(processed_text)

    # Determina o sentimento baseado no compound score
    if scores["compound"] >= 0.05:
        sentiment = "Positivo"
    elif scores["compound"] <= -0.05:
        sentiment = "Negativo"
    else:
        sentiment = "Neutro"

    return {
        "sentiment": sentiment,
        "compound": scores["compound"],
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
    }


# Análise de aspectos
def analyze_aspects(text):
    aspects = {
        "App": ["app", "aplicativo", "tela", "interface", "navegação", "bug"],
        "Crédito": ["credito", "crédito", "limite", "emprestimo", "score"],
        "Atendimento": ["atendimento", "suporte", "chat", "ajuda", "duvida"],
        "Transferência": ["transferencia", "pix", "ted", "pagamento", "saldo"],
        "Taxas": ["taxa", "tarifa", "custo", "cobrança", "juros"],
        "Segurança": ["segurança", "fraude", "golpe", "senha", "bloqueio"],
        "Conta": ["conta", "digital", "cartão", "cadastro", "abertura"],
    }

    text_lower = text.lower()
    found_aspects = defaultdict(int)

    for aspect, keywords in aspects.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_aspects[aspect] += 1

    return dict(found_aspects) if found_aspects else {"Geral": 1}


def main():
    st.title("Análise de Sentimentos - Comentários de Instituições Financeiras")

    # Inicializa o analisador de sentimentos
    sia = setup_nltk()

    # Carrega os dados
    with st.spinner("Carregando dados..."):
        data = get_data_from_db()

    if data is None:
        st.error("Não foi possível carregar os dados.")
        return

    st.write("Aqui estão os dados extraídos do banco de dados:")
    st.write(data)

    # Análise de sentimentos
    with st.spinner("Analisando sentimentos..."):
        sentiments = [analyze_sentiment(comment, sia) for comment in data["comentario"]]

        data["sentiment"] = [s["sentiment"] for s in sentiments]
        data["compound_score"] = [s["compound"] for s in sentiments]
        data["positive_score"] = [s["pos"] for s in sentiments]
        data["negative_score"] = [s["neg"] for s in sentiments]
        data["neutral_score"] = [s["neu"] for s in sentiments]

    # Análise de aspectos
    with st.spinner("Analisando aspectos..."):
        data["aspects"] = [analyze_aspects(comment) for comment in data["comentario"]]

    # Container centralizado para os gráficos
    st.write("## Visualizações dos Resultados")

    # Nuvem de Palavras
    st.write("")
    st.subheader("Nuvem de Palavras dos Comentários")

    # Gera a nuvem de palavras
    wordcloud = generate_wordcloud(data["comentario"])

    # Cria e exibe o gráfico
    fig_wordcloud = plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig_wordcloud)

    # Gráfico de Frequência de Palavras
    st.write("")
    st.subheader("Top 10 Palavras Mais Frequentes")

    # Obtém as palavras mais frequentes
    top_words = get_word_frequency(data["comentario"])

    # Cria o gráfico de barras
    fig_freq = plt.figure(figsize=(12, 6))
    plt.bar(top_words.keys(), top_words.values(), color="skyblue")
    plt.title("Frequência das Palavras Mais Comuns")
    plt.xlabel("Palavras")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")

    # Adiciona os valores sobre as barras
    for i, (word, freq) in enumerate(top_words.items()):
        plt.text(i, freq, str(freq), ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig_freq)

    # Nuvem de Palavras com Pré-processamento
    st.write("")
    st.subheader("Nuvem de Palavras dos Comentários (Com Pré-processamento)")

    # Gera a nuvem de palavras processada
    wordcloud_processed = generate_processed_wordcloud(data["comentario"])

    # Cria e exibe o gráfico
    fig_wordcloud_processed = plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_processed, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig_wordcloud_processed)

    # Gráfico de Frequência de Palavras Processadas
    st.write("")
    st.subheader("Top 10 Palavras Mais Frequentes (Com Pré-processamento)")

    # Obtém as palavras mais frequentes processadas
    top_words_processed = get_processed_word_frequency(data["comentario"])

    # Cria o gráfico de barras
    fig_freq_processed = plt.figure(figsize=(12, 6))
    plt.bar(
        top_words_processed.keys(), top_words_processed.values(), color="lightgreen"
    )
    plt.title("Frequência das Palavras Mais Comuns (Com Pré-processamento)")
    plt.xlabel("Palavras")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")

    # Adiciona os valores sobre as barras
    for i, (word, freq) in enumerate(top_words_processed.items()):
        plt.text(i, freq, str(freq), ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig_freq_processed)

    st.write("")

    # Grafo de Tetragramas
    st.write("")
    st.subheader("Grafo de Tetragramas Mais Frequentes")

    # Cria o grafo
    G, top_tetragrams = create_tetragrams_graph(data["comentario"])

    # Configura o layout do grafo
    pos = nx.spring_layout(G, k=3, seed=20, iterations=50)

    # Cria a figura
    fig_graph = plt.figure(figsize=(15, 10))

    # Desenha as arestas com largura baseada no peso
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        width=[w / max(edge_weights) * 3 for w in edge_weights],
        alpha=0.5,
        arrows=True,
        arrowsize=20,
    )

    # Desenha os nós
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000, alpha=0.7)

    # Adiciona labels aos nós
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.title("Conexões entre Palavras Frequentes nos Comentários", fontsize=20)
    plt.axis("off")
    plt.tight_layout()

    # Mostra o grafo
    st.pyplot(fig_graph)

    st.write("")
    st.subheader("Frequência dos Tetragramas Mais Comuns")

    top_tetragrams_bar = get_tetragrams(data["comentario"], top_n=20)

    # Prepara e ordena os dados para o gráfico
    sorted_tetragrams = sorted(
        top_tetragrams_bar.items(), key=lambda item: item[1]
    )  # Ordenação crescente
    tetagram_labels = [" → ".join(tetagram) for tetagram, _ in sorted_tetragrams]
    frequencies = [freq for _, freq in sorted_tetragrams]

    # Cria o gráfico de barras horizontais
    fig_tetagram = plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(frequencies)), frequencies, color="lightblue")

    # Configurações do gráfico
    plt.title(
        "Tetragramas Mais Frequentes nos Comentários", fontsize=16
    )  # Tamanho do título ajustado
    plt.xlabel("Frequência", fontsize=12)
    plt.ylabel("Sequência de Palavras", fontsize=12)

    # Configura os rótulos do eixo Y
    plt.yticks(range(len(tetagram_labels)), tetagram_labels, fontsize=10)

    # Adiciona os valores ao lado das barras
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=10,
        )

    # Ajusta o layout para evitar cortes
    plt.tight_layout()

    # Mostra o gráfico
    st.pyplot(fig_tetagram)

    ####

    # Nuvem de Palavras com Stopwords Personalizadas
    st.write("")
    st.subheader("Nuvem de Palavras dos Comentários (Com Stopwords Personalizadas)")

    # Gera a nuvem de palavras
    wordcloud_custom = generate_custom_wordcloud(data["comentario"])

    # Cria e exibe o gráfico
    fig_wordcloud_custom = plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_custom, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig_wordcloud_custom)

    # Gráfico de Frequência de Palavras
    st.write("")
    st.subheader("Top 10 Palavras Mais Frequentes (Com Stopwords Personalizadas)")

    # Obtém as palavras mais frequentes
    top_words_custom = get_custom_word_frequency(data["comentario"])

    # Cria o gráfico de barras
    fig_freq_custom = plt.figure(figsize=(12, 6))
    plt.bar(top_words_custom.keys(), top_words_custom.values(), color="lightgreen")
    plt.title("Frequência das Palavras Mais Comuns")
    plt.xlabel("Palavras")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha="right")

    # Adiciona os valores sobre as barras
    for i, (word, freq) in enumerate(top_words_custom.items()):
        plt.text(i, freq, str(freq), ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig_freq_custom)

    ###

    # Dentro da função main(), após as outras visualizações
    st.write("")
    st.subheader("Análise Detalhada dos Comentários")

    # Cria DataFrame com análise detalhada
    detailed_analysis = []
    for _, row in data.iterrows():
        # Pega o comentário original
        original = row["comentario"]

        # Processa o comentário com as stopwords personalizadas
        processed = preprocess_text_with_custom_stopwords(original)

        # Pega o sentimento já calculado
        sentiment = row["sentiment"]
        compound_score = row["compound_score"]
        positive_score = row["positive_score"]
        negative_score = row["negative_score"]
        neutral_score = row["neutral_score"]

        detailed_analysis.append(
            {
                "Comentário Original": original,
                "Comentário Processado": processed,
                "Sentimento": sentiment,
                "Score Compound": f"{compound_score:.3f}",
                "Score Positivo": f"{positive_score:.3f}",
                "Score Negativo": f"{negative_score:.3f}",
                "Score Neutro": f"{neutral_score:.3f}",
            }
        )

    # Cria DataFrame com a análise
    df_analysis = pd.DataFrame(detailed_analysis)

    # Aplica estilo para destacar os sentimentos com cores
    def highlight_sentiment(val):
        if val == "Positivo":
            return "background-color: #90EE90"  # Verde claro
        elif val == "Negativo":
            return "background-color: #ec231a"  # Vermelho claro
        else:
            return "background-color: #6d6767"  # Cinza claro

    def highlight_score(val):
        try:
            score = float(val)
            if score > 0.5:
                return "background-color: #57aef2"  # Verde muito claro
            elif score < -0.5:
                return "background-color: #57aef2"  # Vermelho muito claro
            else:
                return "background-color: #57aef2"  # Cinza muito claro
        except:
            return ""

    # Aplica o estilo e mostra a tabela
    styled_df = df_analysis.style.applymap(
        highlight_sentiment, subset=["Sentimento"]
    ).applymap(
        highlight_score,
        subset=["Score Compound", "Score Positivo", "Score Negativo", "Score Neutro"],
    )

    st.dataframe(styled_df, use_container_width=True)

    # Adiciona filtros
    st.write("")
    st.subheader("Filtros")

    col1, col2 = st.columns(2)
    with col1:
        sentiment_filter = st.selectbox(
            "Filtrar por Sentimento:", ["Todos", "Positivo", "Neutro", "Negativo"]
        )

    with col2:
        score_filter = st.selectbox(
            "Ordenar por Score:", ["Compound", "Positivo", "Negativo", "Neutro"]
        )

    # Aplica filtros
    if sentiment_filter != "Todos":
        filtered_df = df_analysis[df_analysis["Sentimento"] == sentiment_filter]
    else:
        filtered_df = df_analysis

    # Ordena por score selecionado
    score_column_map = {
        "Compound": "Score Compound",
        "Positivo": "Score Positivo",
        "Negativo": "Score Negativo",
        "Neutro": "Score Neutro",
    }

    filtered_df = filtered_df.sort_values(
        by=score_column_map[score_filter],
        ascending=False,
        key=lambda x: x.astype(float),
    )

    if sentiment_filter != "Todos":
        st.write(
            f"Mostrando comentários {sentiment_filter}s ordenados por Score {score_filter}:"
        )
        styled_filtered_df = filtered_df.style.applymap(
            highlight_sentiment, subset=["Sentimento"]
        ).applymap(
            highlight_score,
            subset=[
                "Score Compound",
                "Score Positivo",
                "Score Negativo",
                "Score Neutro",
            ],
        )
        st.dataframe(styled_filtered_df, use_container_width=True)

    ######

    st.write("")
    # Primeiro gráfico - Distribuição de Sentimentos
    st.subheader("Distribuição de Sentimentos")
    sentiment_counts = data["sentiment"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = {"Positivo": "green", "Neutro": "gray", "Negativo": "red"}
    bars = sentiment_counts.plot(
        kind="bar", ax=ax1, color=[colors[x] for x in sentiment_counts.index]
    )

    plt.title("Distribuição dos Sentimentos")
    plt.xlabel("Sentimento")
    plt.ylabel("Número de Comentários")

    for i, v in enumerate(sentiment_counts):
        ax1.text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig1)

    # Espaço entre os gráficos
    st.write("")

    # Segundo gráfico - Distribuição de Scores
    st.subheader("Distribuição de Scores")
    fig2 = plt.figure(figsize=(10, 6))
    plt.hist(data["compound_score"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribuição dos Scores de Sentimento")
    plt.xlabel("Score Composto")
    plt.ylabel("Frequência")
    plt.tight_layout()
    st.pyplot(fig2)

    # Espaço entre os gráficos
    st.write("")

    # Terceiro gráfico - Análise de Aspectos
    st.subheader("Análise de Aspectos")
    all_aspects = defaultdict(int)
    for aspects in data["aspects"]:
        for aspect, count in aspects.items():
            all_aspects[aspect] += count

    fig3 = plt.figure(figsize=(10, 6))
    aspect_items = sorted(all_aspects.items(), key=lambda x: x[1], reverse=True)
    aspects, counts = zip(*aspect_items)

    plt.bar(aspects, counts, color="lightblue")
    plt.title("Aspectos Mencionados nos Comentários")
    plt.xlabel("Aspectos")
    plt.ylabel("Número de Menções")
    plt.xticks(rotation=45)

    # Adiciona os valores sobre as barras
    for i, v in enumerate(counts):
        plt.text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    st.pyplot(fig3)

    # Métricas em uma única linha
    st.write("")
    st.subheader("Métricas Principais")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score Médio", f"{data['compound_score'].mean():.3f}")
    with col2:
        st.metric("Score Máximo", f"{data['compound_score'].max():.3f}")
    with col3:
        st.metric("Score Mínimo", f"{data['compound_score'].min():.3f}")

    # Estatísticas gerais
    st.write("")
    total_comments = len(data)
    positive_perc = (data["sentiment"] == "Positivo").mean() * 100
    negative_perc = (data["sentiment"] == "Negativo").mean() * 100
    neutral_perc = (data["sentiment"] == "Neutro").mean() * 100

    st.write(
        """
    ### Estatísticas Gerais:
    - Total de comentários analisados: {}
    - Porcentagem de comentários positivos: {:.1f}%
    - Porcentagem de comentários negativos: {:.1f}%
    - Porcentagem de comentários neutros: {:.1f}%
    """.format(
            total_comments, positive_perc, negative_perc, neutral_perc
        )
    )

    # Exemplos de comentários
    st.write("")
    st.subheader("Exemplos de Comentários por Categoria")
    for sentiment in ["Positivo", "Neutro", "Negativo"]:
        st.write(f"\n### Comentários {sentiment}s:")
        examples = data[data["sentiment"] == sentiment]["comentario"].head(3)
        for i, example in enumerate(examples, 1):
            st.write(f"{i}. {example}")


if __name__ == "__main__":
    main()

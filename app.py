import streamlit as st
import functions as ushmm
from gensim.models import Word2Vec
import glob
import pandas as pd
import fasttext
from gensim.models import KeyedVectors



def main(model):
    st.markdown(
            f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 1500px;
        }}
    </style>
    """,
            unsafe_allow_html=True,
        )
    texts, irns = load_corpus()
    files = texts[1]
    final_files = []
    for file in files:
        file = file[0]
        if file not in final_files:
            final_files.append(file)

    st.title('Bitter Aloe Project Word Embedding Search')
    st.write("Welcome to the BAP Word Embedding Search Tool. This search engine is designed to allow for you to search for multiple words at the same time through a word embedding algorithm. The idea is to capture abstraction with a single search across documents. Values for each document are determined by proximity and depth in word embeddings relative to the initial search term.")
    st.write("On the left, you can run your search. First, select rather you want to do a segment search (within all documents) or a document search (values are calculated by document). Next, write a word you wish to search for. Ideally, this should be something abstract, like 'hunger'. Next, select your limiting words. This ensures no false positives. For a first pass, always write the initial search term. Finally enter your tier design. This will affect how the net is cast into the word embeddings.")

    st.sidebar.image("images/bap_logo.png")
    form = st.sidebar.form("Options")
    style_option = form.selectbox("How do you want to calculate values?", ("Segment", "Document"))
    keyword = form.text_input('Enter your key search word here:')
    limiting_words = form.text_input('Enter your limiting words here, separated by commas:')
    tiers = form.text_input('Enter your tier design here, separated with commas:')
    removal_words = form.text_input('Enter your optional removal words, separated with commas:')
    search = form.form_submit_button("Search")


    if search:
        final_limiting_words = []
        words = limiting_words.split(",")
        for word in words:
            word = word.strip()
            final_limiting_words.append(word)


        tiers = tiers.split(",")
        final_tiers = []
        for tier in tiers:
            tier = int(tier)
            final_tiers.append(tier)

        removal_words = removal_words.split(",")
        final_removal_words = []
        for word in removal_words:
            word = word.strip()
            final_removal_words.append(word)


        res = ushmm.run_algo(keyword, model, texts, final_limiting_words, style_option, tiers=final_tiers, removal_words=final_removal_words)

        wordvals = res["word_vals"]
        words = wordvals.keys()
        vals = wordvals.values()
        results = res["results"]
        st.sidebar.write(f"These are the words caught by your parameters with their corresponding value relative to {keyword}")
        st.sidebar.write(pd.DataFrame({
                                'Words': words,
                                'Values': vals
                                }))
        rg = []
        texts = []
        values = []
        transcripts = []
        train_data = []

        for x in results:
            value, text, other = x
            values.append(int(value))
            rg_num = text[0].replace("ocr/", "").replace(".txt", "")
            final_text = text[1].replace("A: ", "<br>A: ").replace("Q: ", "<br>Q: ")
            if final_text[:4] == "<br>":
                final_text = final_text[4:]
            texts.append(final_text)
            transcript = f'{rg_num}'
            rg.append(transcript)
            train_data.append((text[1], keyword))
        st.write(f'There are {len(values)} results based on your parameters...')

        table_data = pd.DataFrame({
                                'Value': values,
                                'Testimony': rg,
                                'Extracted Text': texts
                                })
        table_data = table_data.to_html(escape=False)
        st.write(table_data, unsafe_allow_html=True)

        st.write("Here is the data as a JSON file:")
        st.write(res)
        st.write("Here is sample training data as a JSON file:")
        st.write(train_data)
#Cache the model in memory
@st.cache(allow_output_mutation=True)
def load_model():
    model = KeyedVectors.load_word2vec_format("model/bap_15.vec", unicode_errors='ignore')
    return model

@st.cache(allow_output_mutation=True)
def load_corpus():
    texts = ushmm.gen_corpus("ocr/*txt")
    irns = []
    return texts, irns

texts = load_corpus()
files = texts[1]
model = load_model()

if __name__ == "__main__":
    main(model)

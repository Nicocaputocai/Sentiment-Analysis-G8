import tweepy as tw
import pandas as pd
import joblib
import streamlit as st
from st_aggrid import AgGrid
import regex as re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stopwords_es = [word for word in stopwords.words('spanish')]
stopwords_es.append('q')
stopwords_es.append('si')

#Claves de Twitter API
consumer_key = 'value'
consumer_secret = 'value'
access_token = 'value'
access_token_secret = 'value'
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

classifier = joblib.load('sentiment-model.pkl') #importamos el modelo
classifier_es = joblib.load('sentiment-model-esp.pkl') #importamos el modelo

#Preprocesamiento de texto

@st.cache(suppress_st_warning=True)
def strip_all_entities_es(text): 
    #text = re.sub(r"(?:\RT?)\S+", "", text) #removes retweets
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    #text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

#Remove punctuations, links, mentions and \r\n new line characters
@st.cache(suppress_st_warning=True)
def strip_all_entities(text): 
    #text = re.sub(r"(?:\RT?)\S+", "", text) #removes retweets
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
@st.cache(suppress_st_warning=True)
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

#Filter special characters such as & and $ present in some words
@st.cache(suppress_st_warning=True)
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

@st.cache(suppress_st_warning=True)
def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)

#Título de la página
st.title('Sentiment Analysis G8 DigitalHouse')
st.markdown('**:milky_way: [Tomás Korenblit](https://www.linkedin.com/in/tomas-pablo-korenblit/) | :brain: [Néstor Javier Blanco](https://www.linkedin.com/in/javier-blanco-45a8b914//) | :martial_arts_uniform: [Florencia Cerutti](https://www.linkedin.com/in/mariaflorenciacerutti) | :seedling: [Nicolás Caputo](https://www.linkedin.com/in/pablo-nicolas-caputo/)**', unsafe_allow_html=True)


# Arranca la parte ejecutable
def run():

    lang = st.radio(
        "Change model language:",
        ('English (🇺🇸 🇬🇧)', 'Spanish (🇦🇷)'))

    if lang == 'Spanish (🇦🇷)':

        #Aclaración de como funciona
        st.markdown(
            'Esta app usa tweepy para conseguir tweets de twitter basado en el nombre / frase. Entonces procesa los tweets a través de nuestro modelo para sentiment analysis. Los resultados de sentimientos y correspondientes tweets son entonces puestos en un dataframe el cual ven como resultado.')

        with st.form(key='Introducí nombre'):
            search_words = st.text_input(
                'Introducí el nombre para el que querés saber el sentimiento.') #Campo para poner lo que queremos buscar
            number_of_tweets = st.number_input(
                'Introducí el numero para la cantidad de tweets recientes de los cuales querés saber el sentimiento (Maximo 100 tweets)',
                1, 100, 10) #Cantidad de tweets que vamos a traer
            submit_button = st.form_submit_button(label='Enviar')
            
            if submit_button:
                tweets = tw.Cursor(api.search_tweets, q=search_words + " -filter:retweets",
                                lang='es', tweet_mode='extended').items(number_of_tweets) #Cuando se apreta el boton submit, busca lo que le pedimos
                tweet_list = [i.full_text for i in tweets] #Lista de tweets 

                texts_new = [] #Variable nueva para los tweets limpios
                for t in tweet_list:
                    texts_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities_es(t)))))

                df = pd.DataFrame(list(zip(texts_new, classifier_es.predict((texts_new)))),
                                columns=[
                                    'Latest ' + str(number_of_tweets) +
                                    ' Tweets' + ' on ' + search_words,
                                    ' Sentimiento'
                                ]
                                )
                
                # Mostrar dataframe tweet | sentimiento
                AgGrid(df, fit_columns_on_grid_load=True, height=400)
    
    
                # Grafico contar sentimientos
                fig_count = plt.figure()
                sns.countplot(x=df[' Sentimiento'], palette='plasma')
                plt.show()
                st.pyplot(fig_count)

                df['tweets'] = texts_new

                # Grafico wordcloud

                # Read the whole text.
                allWords = ' '.join( [twts for twts in df['tweets']] )

                # Create and generate a word cloud image:
                wordcloud = WordCloud(stopwords=stopwords_es,background_color='white',colormap='plasma',width=1600, height=800).generate(allWords)

                fig_wordcloud = plt.figure()
                # Display the generated image:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.show()
                st.pyplot(fig_wordcloud, dpi=300)
        
        st.markdown('Esta app analiza el sentimiento detrás de una oración.')
        #Aca volvemos a hacer lo mismo pero para predicir un tweet escrito a mano por nosotros

        with st.form(key='Escribí una oración'):
            sentence = st.text_input('Escribí una oración para la que quieras saber el sentimiento')
            submit_button_type = st.form_submit_button(label='Enviar')
            
            if submit_button_type:

                result_new = []
                for t in [sentence]:
                    result_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities_es(t)))))


                result = classifier_es.predict((result_new))
                #Aca le decimos como mostrar los resultados de la predicción

                if result[0] == 'NEG':
                    st.error('Negativo')
                elif result[0] == 'POS':
                    st.success('Positivo')
                else:
                    st.write('algo extraño sucedió:')
                    st.write(result[0])
    
    else:
        #Aclaración de como funciona
        st.markdown(
            'This app uses tweepy to get tweets from twitter based on the input name/phrase. It then processes the tweets through our model for sentiment analysis. The resulting sentiments and corresponding tweets are then put in a dataframe for display which is what you see as result.'
        )

        with st.form(key='Enter name'):
            search_words = st.text_input(
                'Enter the name for which you want to know the sentiment') #Campo para poner lo que queremos buscar
            number_of_tweets = st.number_input(
                'Enter the number of latest tweets for which you want to know the sentiment(Maximum 100 tweets)',
                1, 100, 10) #Cantidad de tweets que vamos a traer
            submit_button = st.form_submit_button(label='Submit')
            if submit_button:
            
                tweets = tw.Cursor(api.search_tweets, q=search_words + " -filter:retweets",
                                lang='en', tweet_mode='extended').items(number_of_tweets) #Cuando se apreta el boton submit, busca lo que le pedimos
                tweet_list = [i.full_text for i in tweets] #Lista de tweets 

                texts_new = [] #Variable nueva para los tweets limpios
                for t in tweet_list:
                    texts_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(t)))))

                df = pd.DataFrame(list(zip(texts_new, classifier.predict((texts_new)))),
                                columns=[
                                    'Latest ' + str(number_of_tweets) +
                                    ' Tweets' + ' on ' + search_words,
                                    ' Sentiment'
                                ]
                                )
                
                # Mostrar dataframe tweet | sentimiento
                AgGrid(df, fit_columns_on_grid_load=True, height=400)

                # Grafico contar sentimientos
                fig_count = plt.figure()
                sns.countplot(x=df[' Sentiment'], palette='plasma')
                plt.show()
                st.pyplot(fig_count)

                df['tweets'] = texts_new

                # Grafico wordcloud

                # Read the whole text.
                allWords = ' '.join( [twts for twts in df['tweets']] )

                # Create and generate a word cloud image:
                wordcloud = WordCloud(background_color='white',colormap='plasma',width=1600, height=800).generate(allWords)

                fig_wordcloud = plt.figure()
                # Display the generated image:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.show()
                st.pyplot(fig_wordcloud, dpi=300)
        
        st.markdown('This app analyses the sentiment behind a sentence.')
        #Aca volvemos a hacer lo mismo pero para predicir un tweet escrito a mano por nosotros

        with st.form(key='Type sentence'):
            sentence = st.text_input('Type a sentence for which you want to know the sentiment')
            submit_button_type = st.form_submit_button(label='Submit')
            
            if submit_button_type:

                result_new = []
                for t in [sentence]:
                    result_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(t)))))


                result = classifier.predict((result_new))
                #Aca le decimos como mostrar los resultados de la predicción

                if result[0] == 'NEG':
                    st.error('Negative')
                elif result[0] == 'POS':
                    st.success('Positive')

if __name__ == '__main__':
    run()
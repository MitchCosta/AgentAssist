import numpy as np
import pandas as pd

import re
import joblib
from utils import clean_tokenize, match_vocabulay

from ast import literal_eval

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score

from collections import Counter

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def get_tf_idf_score(tf_idf_arr, tf_idf_arr_sentence):

    # find the questions in faq with greater similarity with the total received tokenized

    np_dot_score_list = []
    for n in range(0, tf_idf_arr.shape[0]):
        np_dot_score_list.append(np.dot(tf_idf_arr[n], tf_idf_arr_sentence[0]))

    np_dot_df = pd.DataFrame(np_dot_score_list, columns=['value'])  

    np_dot_df = np_dot_df.assign(tokenized_question=tokenized_questions.values)
    np_dot_df = np_dot_df.assign(question=cisco_data['answer_title'][tokenized_questions.index].values)

    return np_dot_df.sort_values('value', ascending=False).head(5)


def predict_1(sentence_one):

    tf_idf_sentece = tf_idf_vectorizer.transform(sentence_one)
    tf_idf_array_sentence = tf_idf_sentece.toarray()

    # DataFrame with candidate questions ordered by similarity (dot product)
    questions_candidates = get_tf_idf_score(tf_idf_array, tf_idf_array_sentence)

    print(questions_candidates)
    return questions_candidates

def predict_2(questions_candidates):

    output_dict_list = []

    for row in range(questions_candidates.shape[0]):

        QA_input = {
            'question': cisco_data['answer_title'][questions_candidates.index[row]],
            'context': cisco_data['answer_paragraphs'][questions_candidates.index[row]]
        }

        model_answer = nlp(QA_input)

        output_dict = {
            'question_index': int(questions_candidates.index[row]),
            'question_value': float(questions_candidates.value[questions_candidates.index[row]]),
            'question': cisco_data['question_original'][questions_candidates.index[row]],
            'model_answer_value': float(model_answer['score']),
            'model_answer': model_answer['answer']
        }
        output_dict_list.append(output_dict)

    print(output_dict_list)
    return output_dict_list


data_path = '../data/'

# START/START  SAME CODE AS IN SPEECH - - generate questions vocabulary
cisco_data = pd.read_csv(data_path + 'cisco_faq_cleaned.csv')

stop1 = pd.read_csv(data_path + 'stop_words.csv', names=['word'])
stop_words = stop1['word'].values.tolist()

cisco_data.drop_duplicates(subset=['answer_title'], inplace=True)
cisco_data = cisco_data.reset_index(drop=True)

# save the original question for use with the model
cisco_data['question_original'] = cisco_data['answer_title']

# join the list of paragraphs in one string
cisco_data['answer_paragraphs'] = cisco_data['answer_paragraphs'].apply(literal_eval)
cisco_data['answer_paragraphs'] = cisco_data['answer_paragraphs'].apply(lambda x: ' '.join(x))
cisco_data['answer_paragraphs'] = cisco_data['answer_paragraphs'].apply(lambda x: x.replace('\xa0', ' '))

# prepare the questions for pre-process
cisco_data['answer_title'] = cisco_data['answer_title'].apply(lambda s:s.lower() if type(s) == str else s)
cisco_data['answer_title'] = cisco_data['answer_title'].apply(lambda x: re.sub('[^a-z0-9.]', ' ', x))
cisco_data['answer_title'] = cisco_data['answer_title'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))

tokenized_questions = cisco_data['answer_title'].apply(lambda x: x.split(' '))
tokenized_questions = tokenized_questions.apply(lambda x:[word for word in x if not word in stop_words])

# tf-idf initializer
MAX_FEATURES = 1000
list_tokenized_questions = tokenized_questions.tolist()

outlst = [' '.join([str(c) for c in lst]) for lst in list_tokenized_questions]

tf_idf_vectorizer = TfidfVectorizer()
tf_idf_train = tf_idf_vectorizer.fit_transform(outlst)

tf_idf_array = tf_idf_train.toarray()
# END/END  SAME CODE AS IN SPEECH - - generate questions vocabulary



# SESSION STARTS HERE
session_dict = {}

# pre-process new incoming text
received_text = 'Hello to you. How is the weather in New York mate? speaker selection via browser was removed from webRTC app'

tokenized_received = clean_tokenize(received_text, stop_words)
tokenized_received = match_vocabulay(tokenized_received, tf_idf_vectorizer)

#   add token to current session
for word in tokenized_received:
    if word not in session_dict:
        session_dict[word] = 1
    else:
        session_dict[word] += 1

# create a sentence with the top 10 most frequent tokens ---- SUBJECT TO CHANGES
# This is the sentence to be feed the tf-idf
sentence_one = [" ".join(dict(Counter(session_dict).most_common(10)))]

print('TOKENS from sentence_one: ', sentence_one)


#  T H E   M O D E L   G O E S   H E R E 

#model_name = "deepset/roberta-base-squad2"
model_name = 'roberta-base-squad2.joblib'

print('model', model_name, 'is loading...')
#nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
nlp = joblib.load(model_name)
print('model loaded!!!')

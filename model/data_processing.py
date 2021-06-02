import re
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer

# Import data from "questions.txt" and "answers.txt".
# These files contain about 15000 random questions and corresponding answers
# obtained from the original dataset.
# To figure out how these two files were created, refer to
questions = []
answers = []
# open file and read the content in a list
with open('questions.txt', 'r') as file_q:
    for question_2 in file_q:
        # remove linebreak which is the last character of the string
        currentPlace = question_2[:-1]

        # add item to the list
        questions.append(currentPlace)
with open('answers.txt', 'r') as file_a:
    for answer_2 in file_a:
        # remove linebreak which is the last character of the string
        currentPlace = answer_2[:-1]

        # add item to the list
        answers.append(currentPlace)


# Only take sentences that are at most 20-word long.
sorted_ques = []
sorted_ans = []
for i in range(len(questions)):
    if len(questions[i]) <= 20:
        sorted_ques.append(questions[i])
        sorted_ans.append(answers[i])

del(questions, answers)


def clean_text(txt):
    # This function process special characters
    # and abbreviations in the original text
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    txt = re.sub(r"n't", " not", txt)
    txt = re.sub(r"n'", "ng", txt)
    txt = re.sub(r"'bout", "about", txt)
    txt = re.sub(r"'til", "until", txt)
    txt = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,']", "", txt)
    return txt


clean_ques = []
clean_ans = []
for line in sorted_ques:
    clean_ques.append(clean_text(line))

# Add <START> token to answers to initiate the prediction process
# Add <END> token to the answers to signify the end of sentence.
for line in sorted_ans:
    clean_ans.append('<START> '
                     + clean_text(line)
                     + ' <END>')


del(sorted_ans, sorted_ques)

# Tokenization
target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'
tokenizer = Tokenizer(filters=target_regex)
tokenizer.fit_on_texts(clean_ques + clean_ans)
VOCAB_SIZE = len(tokenizer.word_index) + 1
print('Vocabulary size : {}'.format(VOCAB_SIZE))

# Convert each question sentence into a sequence of integers
tokenized_questions = tokenizer.texts_to_sequences(clean_ques)
maxlen_questions = 20
# Padding
encoder_input_data = pad_sequences(tokenized_questions,
                                   maxlen=maxlen_questions,
                                   padding='post')

# Convert each answer sentence into a sequence of integers
tokenized_answers = tokenizer.texts_to_sequences(clean_ans)
maxlen_answers = 20
# Padding
decoder_input_data = pad_sequences(tokenized_answers,
                                   maxlen=maxlen_answers,
                                   padding='post')

del(clean_ans, clean_ques)


# remove the first 'start' token from every answer since
# the output for the decoder does not need that
for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]


# Padding
padded_answers = pad_sequences(tokenized_answers,
                               maxlen=maxlen_answers,
                               padding='post')

# Create one-hot encoded answers
del(tokenized_answers, tokenized_questions)
decoder_output_data = to_categorical(padded_answers, VOCAB_SIZE)

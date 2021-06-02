# This file shows how the two data files "questions.txt" and "answers.txt"
# were created.
from random import sample, seed


# Import all movie lines from original data source
lines_data = open(
    'cornell-movie-dialogs-corpus/movie_lines.txt',
    encoding='utf-8',
    errors='ignore').read().split('\n')

# Import data on which lines belong to the same conversation
conversations_data = open(
    'cornell-movie-dialogs-corpus/movie_conversations.txt',
    encoding='utf-8',
    errors='ignore').read().split('\n')

# Create a list of conversations, were each item is a list of all the lines ID
# in the same conversation.
conversations = []
for conversation_data in conversations_data:
    conversations.append(
        conversation_data.split(' +++$+++ ')[-1][1:-1].
        replace("'", " ").replace(",", "").split())

# Create a dictionary of movie lines were the key is the ID of the line
# and the value is a string of the content of the line.
lines = {}
for line in lines_data:
    lines[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1]

# Get a random sample of 20000 conversations
seed(16)
conversation_id = sample(conversations, 20000)

questions = []
answers = []

# Create two lists, one for the questions, and the other for the answers to the
# question of the same index.
for conversation in conversation_id:
    for i in range(len(conversation) - 1):
        questions.append(lines[conversation[i]])
        answers.append(lines[conversation[i + 1]])

print(questions[1650])
print(answers[1650])

# Save data to txt files
with open('questions_2.txt', 'w') as qfile:
    for question in questions:
        qfile.write('%s\n' % question)

with open('answers_2.txt', 'w') as afile:
    for answer in answers:
        afile.write('%s\n' % answer)

print(questions[1650])
print(answers[1650])

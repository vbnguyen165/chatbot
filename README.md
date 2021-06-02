## Instructions

### Files necessary for the quick execution of the software are located in the "model" folder:

- **model/rnn_chatbot.py**: main file, containing the implementation of the training
 and testing of the chatbot. **Run this file to use the chatbot**
- model/questions.txt: data used by the model
- model/answers.txt: data used by the model
- model/data_processing.py: the cleaning process of the data, used by the main file
- model/embedding_matrix.dat: contains the embedding matrix, used by the main file
- model/chatbot_wo_attention.h5: weights of the model, used by the main file

### Supplementary files are located in the "supplementary" folder
These files explain the creation of some files in the "model" folder.

- supplementary/import_data.py: shows how the two data files "questions.txt" and "answers.txt" are created
- supplementary/cornell-movie-dialogs-corpus: folder contains original data files, used by import_data.py
- supplementary/embedding.py: shows how the embedding matrix was created, needs 2 files, a duplicate of "data_processing.py" in the "model" folder and another file downloaded from the internet (> 2Gb). It is recommended to only read the file to understand the process rather than actually running it.

### Others
- requirements.txt: required packages for the software

## Sample Responses
![](demonstration.jpg)

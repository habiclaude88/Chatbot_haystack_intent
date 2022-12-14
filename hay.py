#Get a new dataset
import pandas as pd
import haystack


import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


# In-Memory Document Store
from haystack.document_stores import InMemoryDocumentStore

#Store the text on the RAM Memory
document_store = InMemoryDocumentStore()



df = pd.read_csv('https://raw.githubusercontent.com/Ngabo-bajo/NLP-FELLOWSHIP/main/jobinRwanda.csv')

df.iloc[1]['Job Full Info'].split('\n')

job_dir = 'jobs'

df_len = len(df) #Counts of rows in the Dataframe
for index in range(df_len):
  with open(f'{job_dir}/job_{index}.txt', 'w+') as file:
    file.writelines(df.iloc[index]['Job Full Info'].split('.'))


from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http


#Download the data and clean it
docs = convert_files_to_docs(dir_path=job_dir, clean_func=clean_wiki_text, split_paragraphs=True)


#Store the data into the database
document_store.write_documents(docs)

# An in-memory TfidfRetriever based on Pandas dataframes
from haystack.nodes import TfidfRetriever

retriever = TfidfRetriever(document_store=document_store)

from haystack.nodes import FARMReader

#Download and initiate the model
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)


#Bundle together the model and retriver(Which is the extractor)
from haystack.pipelines import ExtractiveQAPipeline

pipe = ExtractiveQAPipeline(reader, retriever)



def get_answer(q):
  prediction = pipe.run(
      query=q, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 1}}
  )
  print(prediction)
  return prediction



from easynmt import EasyNMT

model = EasyNMT('OPUS-MT')



from langdetect import detect

raw_query = "Was ist die vollständige Form von BRD?"
lang = detect(raw_query)
print(lang)


def answering(qs):
    # detecting the language
    raw_query = qs
    lang = detect(qs) #DETECT LANGUAGE

    # translating the question
    question = model.translate(raw_query, source_lang=lang, target_lang='en')

    # get the answer
    response = get_answer(question) 

    answer = response['answers'][0].to_dict()['answer']

    # translate back to original language
    answer = model.translate(answer, source_lang='en', target_lang=lang)
    # return answer
    if response['answers'][0].to_dict()['score'] <= 0.6:
      answer = 'No Answer'

      answer = model.translate(answer, source_lang='en', target_lang=lang)


    return answer
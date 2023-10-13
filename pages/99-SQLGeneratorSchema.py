import json
import boto3

import sqlalchemy
from sqlalchemy import create_engine
# from snowflake.sqlalchemy import URL

from langchain.docstore.document import Document
from langchain import PromptTemplate,SagemakerEndpoint,SQLDatabase, SQLDatabaseChain, LLMChain
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import SQLDatabaseSequentialChain

from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatAnthropic
from langchain.chains.api import open_meteo_docs
from langchain.memory import ConversationBufferMemory

from typing import Dict
import time

import boto3
import streamlit as st
from streamlit_chat import message

glue_databucket_name = 'sagemaker-studio-741094476554-9zkt2s8krvb' #Create this bucket in S3
glue_db_name='ihmnick-bankadditional'
glue_role=  'ihmnick-AWSGlueServiceRole-glueworkshop120'
glue_crawler_name=glue_db_name+'-crawler120'


client = boto3.client('glue')
region=client.meta.region_name


# Connect to S3 using Athena
connathena=f"athena.{region}.amazonaws.com" 
portathena='443' #Update, if port is different
schemaathena=glue_db_name #from user defined params
s3stagingathena=f's3://{glue_databucket_name}/athenaresults/'#from cfn params
wkgrpathena='primary'#Update, if workgroup is different
# tablesathena=['dataset']#[<tabe name>]

# Create the athena connection string
connection_string = f"awsathena+rest://@{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/&work_group={wkgrpathena}"

# Create the athena SQLAlchemy engine
engine_athena = create_engine(connection_string, echo=False)
dbathena = SQLDatabase(engine_athena)
gdc = [schemaathena] 

# Setup memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Generate Dynamic prompts to populate the Glue Data Catalog
# harvest aws crawler metadata
def parse_catalog():
    # Connect to Glue catalog
    # Get metadata of redshift serverless tables
    columns_str=''
    
    # Define glue cient
    glue_client = boto3.client('glue')
    
    for db in gdc:
        response = glue_client.get_tables(DatabaseName =db)
        for tables in response['TableList']:
            #classification in the response for s3 and other databases is different. Set classification based on the response location
            if tables['StorageDescriptor']['Location'].startswith('s3'):  classification='s3' 
            else:  classification = tables['Parameters']['classification']
            for columns in tables['StorageDescriptor']['Columns']:
                    dbname,tblname,colname=tables['DatabaseName'],tables['Name'],columns['Name']
                    columns_str=columns_str+f'\n{classification}|{dbname}|{tblname}|{colname}'                     
    # API
    # Append the metadata of the API to the unified glue data catalog
    columns_str=columns_str+'\n'+('api|meteo|weather|weather')
    print('columns_str', columns_str)
    return columns_str
        
glue_catalog = parse_catalog()

# Function 1 'Infer Channel'
# Define a function that infers the channel/database/table and sets the database for querying
def identify_channel(query):
    db = {}
    
    # Prompt 1 'Infer Channel'
    # Set prompt template. It instructs the llm on how to evaluate and respond to the llm. It is referred to as dynamic since glue data catalog is first getting generated and appended to the prompt.
    prompt_template = """
     From the table below, find the database (in column database) which will contain the data (in corresponding column_names) to answer the question 
     {query} \n
     """+glue_catalog +""" 
     Give your answer as database == 
     Also,give your answer as database.table == 
     """
    
    # Define prompt 1
    PROMPT_channel = PromptTemplate( template=prompt_template, input_variables=["query"]  )

    # define LLM chain
    llm_chain = LLMChain(prompt=PROMPT_channel, llm=llm)
    
    # Run the query and save to generated texts
    generated_texts = llm_chain.run(query)
    print('identified channel:', generated_texts)

    # Set the channel from where the query can be answered
    if 's3' in generated_texts: 
            channel='db'
            db=dbathena
            print("SET database to athena")
    elif 'api' in generated_texts: 
            channel='api'
            print("SET database to weather api")        
    else: 
        raise Exception("User question cannot be answered by any of the channels mentioned in the catalog")
    
    print("Step complete. Channel is: ", channel)
    
    return channel, db

# Define a function that infers the channel/database/table and sets the database for querying
def run_query(query):

    channel, db = identify_channel(query) #call the identify channel function first

    # Prompt 2 'Run Query'
    # After determining the data channel, run the Langchain SQL Database chain to convert 'text to sql' and run the query against the source data channel. 
    # provide rules for running the SQL queries in default template--> table info.

    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.

    Do not append 'Query:' to SQLQuery.
    
    Display SQLResult after the query is run in plain english that users can understand. 

    Provide answer in simple english statement.
 
    Only use the following tables:

    {table_info}

    Question: {input}"""

    PROMPT_sql = PromptTemplate(
        input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
    )

    if channel=='db':
        db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT_sql, verbose=True, return_intermediate_steps=False, memory=memory, use_query_checker=True)
        response = db_chain.run(query)
    elif channel == 'api':
        chain_api = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True, memory=memory)
        response = chain_api.run(query)
    else: 
        raise Exception("Unlisted channel. Check your unified catalog")
    
    return response

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
        
def action_search():
    st.title('Market Research Assistant')
    
    col1, col2 = st.columns(2)
    with col1:
        query = st.text_input('**Ask a question:**', '')
        button_search = st.button('Ask')
        
        if query or button_search:
            reply = run_query(query)
            # store the output 
            st.session_state.past.append(query)
            st.session_state.generated.append(reply)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        
    # col1, col2 = st.columns(2)
    # with col1:
    #     query = st.text_input('**Ask a question:**', '')
    #     button_search = st.button('Ask')
    #     if query or button_search:
    #         message(query, is_user=True)
    #         response = run_query(query)
    #         # st.write(response)
    #         message(response)
                
def app_sidebar():
    with st.sidebar:
        st.write('## How to use:')
        description = """Assume the role of a marketing analyst working at a bank. There is an ask to perform analysis whether a customer will enroll for a certificate of deposit (CD). In order to perform the analysis, the marketing dataset contains information on customer demographics, responses to marketing events, and external factors.

- What percentage of customers in each age group enroll for CDs?
- What is the enrollment rate for each marital status?
- What percentage of customers in each education level enroll for CDs?
- What percentage of customers with existing loans (housing loan, personal loan) enroll for CDs?
- What is the average enrollment rate over time for each month/quarter?
- What is the average enrollment rate at different time periods after the last marketing contact (e.g. 0-7 days, 8-14 days, 15-30 days, 30+ days)?
- What is the enrollment percentage for different outcomes of the previous marketing campaign (e.g. success, failure, no campaign)?
- What percentage of customers contacted X times enroll for CDs, versus customers contacted Y times?"""
        st.write(description)
        st.write('---')


def main():
    st.set_page_config(layout="wide")
    app_sidebar()
    action_search()


if __name__ == '__main__':
    main()
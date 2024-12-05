import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch.nn.functional as F
import joblib
import re

import streamlit as st


import pandas as pd
from io import StringIO

import joblib

import torch
import transformers
import langchain

# from langchain_openai import OpenAI

#When deployed on huggingface spaces, this values has to be passed using Variables & Secrets setting, as shown in the video :)
#import os
#os.environ["OPENAI_API_KEY"] = "sk-PLfFwPq6y24234234234FJ1Uc234234L8hVowXdt"

# from langchain.llms import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceEndpoint
from langchain import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM



#App UI starts here
st.set_page_config(page_title="Drilling guide", page_icon=":robot:")
st.header("Drilling Advisory System Using LLMs")


import os
# api_key = st.sidebar.text_input('Hugging Face API Key:', type='password') 
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# if api_key:
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

# llm_2 = HuggingFaceEndpoint(repo_id="tiiuae/falcon-7b")

# llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3") 

llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct") # Last used model 
 
# llm_2 = HuggingFaceEndpoint(repo_id="google-bert/bert-base-uncased") 

st.sidebar.title("Provide some metadata about the Well 😎")


# Operational info:
st.subheader("Input the Operational info:")
st.sidebar.subheader("Input the Operational info:")

# Wellbore = st.sidebar.text_input("Enter Wellbore name:", key='input_1')
Section_type = st.sidebar.selectbox("Select section type:", ("Move/Mob/Demob", "Drilling", "Intervention", "Completion", "P&A", "Workover"), key='input_2')
Section = st.sidebar.selectbox("Select section:", ('12 1/4"', '16"', '24"','36"'), key='input_3')
# Operation_start_time = st.sidebar.text_input("Enter operation start time:", key='input_4')
# Operation_end_time = st.sidebar.text_input("Enter operation end time:", key='input_5')
Operation_depth = st.sidebar.text_input("Enter operation depth (m) :", key='input_6')

Operation_description = st.text_area("Enter operation description:", key='input_7', height=100)

# Total_time = st.sidebar.text_input("Enter the total time taken to complete the operation (hr) :", key='input_8')
Main_activity = st.sidebar.selectbox("Select the main activity:",('MOVING', 'DRILLING', 'BOP', 'COMPLETION', 'RIG', 'WELL CONTROL', 'CASING', 'Konverteringskode', 'EVALUATION', 'FISHING', 'PLUGBACK', 'WIRELINE', 'TESTING', 'WORKOVER/RECOMPLETION', 'WORKOVER', 'Mob/Demob/Spot/Skid', 'Rig Up/Down', 'Logging CH', 'Rig Over', 'Straddle/Patch', 'Mechanical Plug', 'Other', 'Cutting', 'Pumping', 'Skid Rig', 'Rig/general', 'N/D X-mas Tree', 'Production Riser', 'BOP Test', 'Pull Tubing', 'Pull Tubing Hanger', 'Pull Compl. eq.', 'Drilling Riser', 'Cement', 'Cut Casing', 'Wellhead', 'Pull Casing', 'Clean-out', 'Whipstock/milling', 'Drilling'), key='input_9')
Sub_activity = st.sidebar.selectbox("Select the Sub activity:", ('ANCHOR', 'TRIP', 'TRANSIT', 'BOP', 'EQUIP', 'RIG', 'DRILLING', 'SURVEY', 'CIRC/COND', 'EQUIPMENT', 'SUBEQ', 'CSG/LINER', 'INTERR', 'TEST', 'LOGGING', 'WELLHEAD', 'FISH', 'CORING', 'CEMENT', 'CASING', 'OTHER', 'CIRC', 'RU', 'DOWNHOLE OPERATION.', 'FLOW', 'DOWNHOLE OPERATION..', 'RU/RD', 'SPOTTING', 'RIH ELECT WL', 'FLOWING / INJECTING..', 'StatoilHydro', 'CHANGE BHA', 'POOH ELECT WL', 'RIG OVER', 'MECH TRIP', 'DOWNHOLE OPERATION', 'FLOWING / INJECTING.', 'RD', 'COMPLETION EQUIPMENT.', 'Spotting', 'Other', 'Rig Up', 'Test, Handover', 'M/U, L/D', 'Logging', 'RIH/POOH', 'Change stuffing box/GIH', 'Set/Release', 'Test', 'Rig Down', 'Skid', 'Mob/Demob', 'Inject/Circ/Flow', 'Logging, Injecting', 'Change cable drum', 'Cutting', 'Pumping', 'Maintenance', 'N/D XT', 'Investigation/Inspection', 'N/U BOP', 'Trip', 'Circ/Displace', 'R/U, R/D', 'Pull TH', 'M/U, L/D TH/RT', 'Pull Tubing w/ CL', 'M/U, L/D RT/BHA', 'RIH/POOH w/ RT', 'Release', 'Pull Compl. Eq. on DP', 'L/D Compl. Eq.', 'Pull Tubing', 'N/D BOP', 'Land/Release', 'Run/Pull', 'Cement', 'Cut', 'Lock Down Assembly', 'Seal Assembly', 'Release Casing', 'Pull Csg on RT', 'Pull Casing', 'Wearbushing', 'Clean', 'Install Test Tool', 'Retrieve Test Tool', 'Whipstock/milling'), key='input_10')
Activity = st.sidebar.selectbox("Select the activity:", ('RUN/PULL ANCHORS', 'PICK UP/LAY DOWN DRILL PIPE', 'POSITION RIG/VESSEL', 'RUN/PULL BOP/RISER', 'Establish/Pull guidewires', 'NOT PLANNED RIG MAINTENANCE', 'DRILLING w/MUDMOTOR/PDM', 'SURVEY', 'ROUTINE HOLE CIRC/COND', 'CIRCULATING FOR WELL EVALUATION', 'DRILLING OTHER TIME, OK', 'REPAIR RIG EQUIPMENT', 'Pull Debris Cap/TA Cap', 'HOLE OPEN', 'REAMING TIGHT HOLE', 'RUN CASING', 'DRILL FLOAT/CEMENT', 'WAITING ON WEATHER', 'OTHER + OTHER EQ.( + OTHER(F)', 'LEAK OFF TEST', 'Run in and POOH with BHA', 'Handling of BHA', 'WIPERTRIP INCL. CIRCULATION', 'TRIP OUT FOR LOGGING', 'FORMATION LOGGING WITH WIRELINE', 'RIG UP/DOWN TO RUN CASING', 'PULL/RUN WEARBUSHING, NEG', 'RUN/PULL WEARBUSHING', 'OTHER CASING TIME, OK', 'TESTING CASING/LINER, OK', 'FISHING FOR JUNK/TOOLS', 'TEST SUBSEA BOP, OK', 'CUT CORE', 'TRIP WITH CORE BARREL', 'CIRC/COND', 'LOST CIRC. DURING DRILLING', 'CIRC KILL MUD', 'TRIP TIME ASSOC. WITH P&A', 'SET / TEST CEMENT PLUG', 'CUT/RETREIVE/LAY DOWN CSG', 'OTHER OPERATIONS OK', 'R/p workstring', 'MILLING JUNK', 'Displace to brine', 'SQUEEZE CEMENT', 'RU WL equipment', 'Other waiting.', 'EQUIPMENT TIME OTHER, NEG', 'Install and test Tubing Hanger', 'Install Subsea-XT', 'All downhole correlation and perforating', 'FLOW/CLEAN WELL', 'All electric logging at operating depth', 'OTHER OPERATIONS', 'Other waiting', 'SET / TEST MECHANICAL PLUG', 'Test SCSSV', 'Nipple up XMT', 'Fishing with wire', 'Install Debris Cap/TA Cap', 'Wait on weather', 'Spotting of equipment prior to RU', 'TEST SURFACE BOP, OK', 'RIH EWL to operatn dpth or start tractor', 'Inject well, BHA at surface or stationry', 'Waiting on production dept', 'Change WL BHA', 'All logging operation at operating depth', 'POOH after completed downhole operation', 'Rig over to new cable type', 'RIH/POOH mechanical WL to target depth', 'All work on mech cable excl RIH/POOH', 'POOH electric WL after downhole operatn', 'Pressure/inflow test plug, packers, etc', 'RIH to operation depth or start tractor', 'RD WL equipment', 'WAITING ON CRANE', 'Xmas tree operation, dry wellhead.', 'Spotting equipment', 'Handover', 'R/U equipment', 'Pre-Job Meeting/Brief/Debrief', 'Test DHSV/XT', 'M/U BHA', 'Leak detection log', 'RIH, Standard', 'Evaluate situation', 'L/D BHA', 'Planning operation', 'Change cable/flowtube size', 'RIH, Working', 'Release Straddle/Patch', 'POOH, Working', 'Other', 'Corrosion log', 'POOH, Standard', 'Set Straddle/Patch', 'Test', 'R/D equipment', 'Skid', 'Mobilise equipment', 'Inject/Bullhead', 'Set Plug', 'Demobilise equipment', 'Safe Job Analysis (SJA)', 'Other, Logging', 'Caliper log', 'Release Plug', 'Rebuild cable head', 'Slickline to Braided/Electrical', 'Install 5/16 drum', 'Remove 5/16 drum', 'Braided to Electrical', 'BHA test', 'Circulate', 'Cutting', 'Displace', 'Maintenance - Planned', 'N/D XT', 'Other, Inspection', 'Nipple up BOP', 'Install low pressure riser', 'RIH, Stands', 'Circ/Cond', 'POOH, Stands', 'R/U handling equipment', 'Pull TH, Stands', 'Pull TH, Other', 'L/D TH/RT', 'Pull tubing w/ CL, Joints', 'Pull tubing w/ CL, Other', 'L/D DHSV', 'R/D handling equipment', 'M/U RT/BHA', 'RIH w/ RT, Stands', 'RIH w/ RT, Circ/cond', 'Release', 'Pull Eq. on DP, Stands', 'Pull Eq. on DP, Circ/cond', 'Pull Eq. on DP, Flowcheck', 'L/D Compl. Eq.', 'Pull tubing, Joints', 'Pull tubing, Ch. handl. eq', 'Flowcheck', 'RIH, Other', 'Disconnect low pressure riser', 'Nipple down BOP', 'Release connector', 'L/D Tensioner Spool Piece', 'L/D Conn./Taper/Transition Joint', 'Pull Riser, Joints', 'R/D tensioner system', 'M/U WH conn./ L. Flex Joint', 'Run Riser, Joints', 'M/U Spool/ Centr./ U. Flex Joint', 'Land/lock connector', 'R/U tensioner system', 'Connection test', 'Test surface eq.', 'Function test', 'RIH, Ch. handl. eq', 'POOH, Ch. handl. eq', 'M/U BHA/Stinger', 'RIH, Circ/cond', 'Circ/prep. prior to cementing', 'Cement, Squeeze', 'POOH, Circ/cond', 'Cement, Plug', 'Circ/prep. prior to POOH', 'POOH, Other', 'L/D BHA/Stinger', 'RIH, Wash/Ream', 'POOH, Singles', 'Cut', 'POOH, Flowcheck', 'Retrieve Lock Down Assembly', 'Retrieve Seal Assembly', 'Release casing', 'Pull Csg on RT, Stands', 'Pull Csg on RT, Other', 'Pull Csg on RT, Circ/cond', 'Pull Csg on RT, Flowcheck', 'Pull Csg, Circ/cond', 'Pull Csg on RT, Singles', 'L/D RT/BHA', 'Pull Csg, Joints', 'Pull Csg, Other', 'Install Wearbushing', 'M/U Clean-up Assy', 'Circ/cond', 'Scrape', 'L/D Clean-up Assy', 'Cement log', 'Other, Cement', 'Clean/Displace', 'Install Test Tool', 'Retrieve Test Tool', 'M/U Whipstock/Milling BHA', 'Set Whipstock', 'Mill Window', 'Polish', 'FIT/LOT/XLOT', 'L/D Whipstock/Milling BHA', 'Clean BOP/Riser'), key='input_11')
# Country_name = st.sidebar.text_input("Enter the Country name:", key='input_12')
# Field = st.sidebar.text_input("Enter the Field name:", key='input_13')
# Rig = st.sidebar.selectbox("Enter the Rig name:", ('SCARABEO 5', 'SNORRE A', 'SNORRE RUBICON'), key='input_14')
# Rig_up_method = st.sidebar.selectbox("Enter the Rig up method:",('w/ RIG', 'w/o RIG'), key='input_15')

# Incident info:
st.subheader("Input the incident info:")
st.sidebar.subheader("Input the incident info:")

# Start_time = st.sidebar.text_input("Enter incident start time:", key='input_16')
Incident_type = st.sidebar.selectbox("Select incident type:",('Equipment failure', 'Operation failed'), key='input_17')
Conveyance = st.sidebar.selectbox("Select Conveyance type:", ('Drill Pipe', 'Wireline'), key='input_18')
Service = st.sidebar.selectbox("Select the main Service:", ('Rig Operations', 'Plug', 'Electric Wireline Logging', 'Wireline', 'Compl/sand contr. equip.'), key='input_19')
# Service_code = st.sidebar.selectbox("Enter the Service code:",('RIG', 'PLUG', 'EWL', 'WIR', 'CPLEQ'), key='input_20')
# Company = st.sidebar.selectbox("Enter the company involved:", ('Equinor', 'Archer', 'Halliburton', 'Schlumberger', 'DeepWell'), key='input_21')
# Company_type = st.sidebar.selectbox("Enter the company type:",('Operator', 'Rig contractor', 'Service company'), key='input_22')
# Downtime = st.sidebar.text_input("Enter the Downtime (hr):", key='input_23')
# Downtime_per = st.sidebar.text_input("Enter the Downtime (%):", key='input_24')
# Dev_Cost =  st.sidebar.text_input("Enter the Development cost (NOK):", key='input_25')
Failure_service = st.sidebar.selectbox("Select the Failure service:", ('Rig Operations', 'Plug', 'Electric Wireline Logging', 'Wireline', 'Compl/sand contr. equip.'), key='input_26')
# Failure_code = st.sidebar.selectbox("Enter the Failure code:", ('RIG-E03 BOP stack/valves', 'RIG-E388 General', 'RIG-E06 Deck crane', 'RIG-03 Procedure not followed', 'RIG-01 Procedure', 'RIG-E347 Pipe handling equ/sys other', 'RIG-E341 Vert pipe handling system', 'RIG-E01 BOP choke manifold', 'PLUG-E03 Setting/pulling tool', 'EWL-E02 Auxiliary mechanical', 'WIR-E05 Grease injection/stuffing box', 'WIR-E10 Winch/Power pack', 'CPLEQ-E04 Other equipment', 'WIR-01 Procedure', 'EWL-E04 Cables/Heads & Assoc..'), key='input_27')

# Synergi_title_Eng = st.text_input("Enter the synergy title:", key='input_28')
Description_Eng = st.text_area("Enter the incident description:", key='input_29', height=100)

Equipment_part = st.sidebar.selectbox("Select the equipment part:", ('Kill & choke valve remote', 'Other', 'Control system','CAL-B'), key='input_30')
Equipment_type = st.sidebar.selectbox("Select the equipment type:",('BOP stack', 'Other', 'Gantry crane', 'Catwalk machine', 'Upper racking arm', 'Manifold and line'), key='input_31')

Manufacturer = st.sidebar.selectbox("Select the manufacturer:", ('Shaffer', 'UNKNOWN', 'Maritime Hydraulics',  'Tech Trade'), key='input_32')

# Experience info:
st.subheader("Input the experience info:")
st.sidebar.subheader("Input the experience info:")

Wellbore_classification = st.sidebar.selectbox("Select the Wellbore classification:", ('Injection',), key='input_33')
Well_type = st.sidebar.selectbox("Select the Well type:", ('Development',), key='input_34')
Tight_well_info = st.sidebar.selectbox("Was it a tight well?:", ("Yes", "No", "Not sure"), key='input_35')
Well_installation_type = st.sidebar.selectbox("Select the Well installation type:", ('Platform'), key='input_36')
Discipline = st.sidebar.selectbox("Select the Discipline:", ('Wellsite Geology/Operation Geology', 'Mud Logging ', 'Casing', 'Cementing', 'Subsea WH/X-mas tree', 'Plug', 'Wireline'), key='input_37')
# Company_involved = st.sidebar.selectbox("Enter the company involved:",('Equinor', 'Archer', 'Interwell', 'Halliburton', 'GE Oil and Gas', 'Omega Completion Technology AS', 'DeepWell', 'Schlumberger', 'Aker Kvaerner Well Services'), key='input_38')
# Experience_kind = st.sidebar.selectbox("Enter the experience kind:", ("Positive", "Negative", ""), key='input_39')

Keywords = st.sidebar.selectbox("Select the associated keywords:",('Formation', 'Mud Gas Readings', 'Casing/Liner', 'Cement Plug', 'Cementing', 'Template', 'T/A Plugs & Mech. Plugs', 'Wireline Mechanical', 'Wireline Electrical', 'Straddle'), key='input_40')

# Experience_start_time = st.sidebar.text_input("Enter the experience start time:", key='input_41')
Experience_depth = st.sidebar.text_input("Enter the experience depth (m):", key='input_42')
Mud_loss = st.sidebar.selectbox("Was the mud loss noticed?:", ("Yes", "No", ""), key='input_43')

# Subject = st.text_input("Enter the subject of the experience:", key='input_44')
Experience_description = st.text_area("Enter the experience description:", key='input_45', height=100)

query = f"""
The Wellbore had the Section type: '{Section_type}' and Section: '{Section}'.
The operational depth was { Operation_depth} meters.
The Main activity was '{ Main_activity}' and the Sub activity was '{ Sub_activity}' and the activity was '{Activity}'.

The operational description was: '{ Operation_description}'.

_____________________________________________________________________________________


The incident type was : '{Incident_type}'.
The conveyance was : '{Conveyance}'. The main service was : '{Service}'.
The failure service was : '{Failure_service}'. The equipment part used was : '{Equipment_part}' and equipment type was : '{Equipment_type}'.

The incident description was : '{Description_Eng}'.
_____________________________________________________________________________________

The wellbore was classified as : '{Wellbore_classification}' and  it was of type: '{Well_type}'. Was it a tight well? The answere is : '{Tight_well_info}'. The well installation type was : '{Well_installation_type}'. 
The discipline was : '{Discipline}' and associated keywords are : '{Keywords}'.
This experience was created or started at the depth of : '{Experience_depth} meters'. Was the mud loss noticed? The answer is : '{Mud_loss}'.

The description of the experience was : '{Experience_description}'.
_____________________________________________________________________________________
"""

st.write(query)

                                              # General LLM model:

# input_text = query
# # input_text = st.text_area("Enter the details about the well here: ", key='input', height=100)

# user_input=input_text

# template =  "{our_text} Based on the above text, please provide me top 5 recommendations in the following format: 'Recommendations: <recommendation>'"

# prompt = PromptTemplate(
#     input_variables=["our_text"],
#     template=template)

# final_prompt = prompt.format(our_text=user_input)

# # response = load_answer(user_input)

# submit = st.button('Provide recommendations based on the above text')  

# # If generate button is clicked
# if submit:
    
#     st.subheader("Recommendations:")
#     # text = llm.invoke(final_prompt)
#     # pattern = r'"(.*?)"'
#     # final_text = re.findall(pattern,text)
#     # st.write(final_text[0])

#     text_llm = llm.invoke(final_prompt)

#     if "Recommendations:" in text_llm:
#         generated_text = text_llm.split("Recommendations:", 1)[-1].strip()
    
# # print(generated_text)
#     st.write(generated_text)

#     # st.write(text_llm)

                                              # RAG model:

# st.subheader("Drilling recommendation system using a RAG-LLM model")
                                                                         

import streamlit as st
import torch
from langchain_community.llms import HuggingFaceEndpoint
from langchain import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import os

# api_key = st.sidebar.text_input('Hugging Face API Key:', type='password') 

# Set your Hugging Face API token
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key


# Configure embedding model and settings
Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 1000
Settings.chunk_overlap = 50


# Load LLM and settings
@st.cache_resource
def load_llm():
    # return HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3")
    return HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct")

llm_rag = load_llm()

# Set the number of documents to retrieve
top_k = 1  # Define top_k here

# Load the Vector Index only once
if 'index' not in st.session_state:
    print("Loading Vector Index...")
    #storage_context = StorageContext.from_defaults(persist_dir="VectorStore_rag")
    storage_context = StorageContext.from_defaults(persist_dir="VectorStore_rag")
    st.session_state.index = load_index_from_storage(storage_context=storage_context)
    print("Vector Index loaded successfully.")

# Set up retriever and query engine
if 'query_engine' not in st.session_state:
    top_k = 1
    retriever = VectorIndexRetriever(
        index=st.session_state.index,
        similarity_top_k=top_k,
    )
    
    st.session_state.query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)],
    )

# query = """
# The Wellbore name was : 'NO 34/7-P-28' at Field : 'Snorre' in country :  'Norway', using a Rig called : 'SNORRE A' of rig type: 'Fixed'.
# The wellbore was classified as : 'Injection' and  it was of type: 'Development'. Was it a tight well? The answere is : 'Not tight well'. The well installation type was : 'Platform'. 
# The main section was : 'Prepare Sidetrack' and the section type was: 'P&A'. The Main activity was : 'Pull Casing' and the Sub-activity was : 'Pull Csg on RT' and the activity was 'Pull Csg on RT, Circ/cond'.
# The discipline was : 'Mud Logging ' and the company involved was : 'Equinor'. The experience was a : 'Negative Experience' and associated keywords are : 'Mud Gas Readings'.
# This experience was created or started at : '4/22/2018 10:30:00 PM' and at the depth of : '1201.0 meters'. Was the mud loss noticed? The answer is : 'No'.
# The subject of the experience was : 'High amounts of gas and some oil residue left in 9 5/8" x 13 3/8" casing annulus.'.

# """

if st.button("Provide Top 5 recommendations based on above information", key=12):
    # llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3") 
    # st.write("Model loded successfully")


    print("Finding the similar doc in Vector store...")
    response = st.session_state.query_engine.query(query)

    # Reformat response
    context = "Context:\n - Historical case:\n\n"
    num_retrieved = len(response.source_nodes)  # Get the number of retrieved documents

    print("Preparing the context...")
    for i in range(min(top_k, num_retrieved)):
        node = response.source_nodes[i]
        similarity_score = node.score
        context += f"{node.text}\n\n"  # Add the document text

    # context = context + query
    # st.write("Context:", context)

    # st.write("Similarity Score: ", str(round(similarity_score, 2)) + "%")

    template =  f"{context} Based on the above context, please provide me top 5 recommendations in the following format: 'Recommendations: <recommendation>'"
    prompt = PromptTemplate(input_variables=["context"], template=template)
    final_prompt = prompt.format(context=context)
    # st.write("final_prompt:", final_prompt)
    print(final_prompt)

    st.subheader("Recommendations:")
    final_text = llm_rag.invoke(final_prompt)
    # st.write("Final text:", final_text)
    

    # pattern = r"Survey Name:\s*([A-Za-z0-9-]+)"
    # match = re.search(pattern, final_text)
    # survey_name = match.group(1)
    # st.write(survey_name)

    st.write(final_text)

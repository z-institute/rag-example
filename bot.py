from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, ContextTypes
from telegram.ext.filters import TEXT, COMMAND
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter

import nltk
import os

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key = os.environ.get("OPENAI_API_KEY"))


prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer. For the speaker introduction, be as closely as the original input data.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
llm = ChatOpenAI(temperature=0)

final_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

nltk.download('punkt')

# Define states for the conversation
TRAINING_DATA = 1


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hi! Send me a message and I will find the closest match in the FAISS index.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text
    user_id = str(update.message.from_user.id)
    index_path = f"all_faiss/{user_id}/faiss_index"

    if not os.path.exists(index_path):
        await update.message.reply_text("Sorry, I don't have any data for you yet. Please use the /train command to add some data first.")
        return

    try:
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        ai_response = final_chain.invoke(
            {"question": query, 
             "input_documents": vector_store.similarity_search(query)}
        )
        
        await update.message.reply_text(ai_response["output_text"])
    except Exception as e:
        print(f"Error loading FAISS index or processing query: {e}")
        await update.message.reply_text("Sorry, I encountered an error while processing your request. Please try again later or contact the administrator.")

async def receive_training_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    data = update.message.text
    user_id = str(update.message.from_user.id)
    markdown_file_path = f'train_docs/{user_id}/data.md'

    os.makedirs(os.path.dirname(markdown_file_path), exist_ok=True)

    with open(markdown_file_path, 'a') as file:
        file.write(data + '\n')

    # Place your training logic here
    loader = DirectoryLoader(
    os.path.relpath("train_docs"),
    glob=f'{user_id}/data.md',
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredMarkdownLoader,
    )
    docs = loader.load()

    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', )

    headers_to_split_on = [
        ("#", "Header 1")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = markdown_splitter.split_text(docs[0].page_content)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("all_faiss/"+ user_id + "/faiss_index")
        
    await update.message.reply_text('Trained successfully.')
    return ConversationHandler.END

async def train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text('Please input the training data in markdown format.')
    return TRAINING_DATA

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text('Training canceled.')
    return ConversationHandler.END

def main() -> None:
    application = Application.builder().token(os.environ.get('BOT_TOKEN')).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('train', train)],
        states={
            TRAINING_DATA: [MessageHandler(TEXT & ~COMMAND, receive_training_data)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(TEXT & ~COMMAND, handle_message))
    application.add_handler(CommandHandler("start", start))

    application.run_polling()

if __name__ == '__main__':
    main()
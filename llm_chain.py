from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)
prompt = PromptTemplate(
    
)

chain = LLMChain(llm = llm, prompt = prompt)

if __name__ == '__main__':
    print(chain.run(""))
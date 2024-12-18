from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from backend.config import settings

# TODO: Create a string template for this chain. It must indicate the LLM
# that a resume is being provided to be summarized to extract the candidates skills.
# The template must have one input variables: `resume`.
template = """
You are an AI job finder assistant that extracts relevant skills from a resume. 
The following is a candidate's resume:

{resume}


"""

def get_resume_summarizer_chain():
    # TODO: Create a prompt template using the string template created above.
    # Hint: Use the `langchain.prompts.PromptTemplate` class.
    # Hint: Don't forget to add the input variables: `resume'
    prompt = PromptTemplate(
        input_variables=["resume"], 
        template=template
    )

    # TODO: Create an instance of `langchain.chat_models.ChatOpenAI` with the appropriate
    # settings.
    # Hint: You can use `settings.OPENAI_LLM_MODEL` and `settings.OPENAI_API_KEY`
    # to setup the llm from the project settings.

    llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_LLM_MODEL,  
        temperature=0 
    )
    # TODO: Create an instance of `langchain.chains.LLMChain` with the appropriate settings.
    # This chain must combine our prompt and an llm. It doesn't need a memory.
    resume_summarizer_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )
    
    return resume_summarizer_chain


if __name__ == "__main__":
    resume_summarizer_chain = get_resume_summarizer_chain()
    print(
        resume_summarizer_chain.invoke(
            {"resume": "I am a software engineer with 5 years of experience"}
        )
    )

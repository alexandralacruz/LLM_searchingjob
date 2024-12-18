from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from backend.config import settings


class ChatAssistant:
    def __init__(self, llm_model, api_key, temperature=0, history_length=3):
        """
        Initialize the ChatAssistant class.

        Parameters
        ----------
        llm_model : str
            The model name.

        api_key : str
            The API key for accessing the LLM model.

        temperature : float
            The temperature parameter for generating responses.

        history_length : int, optional
            The length of the conversation history to be stored in memory. Default is 3.
        """
        # TODO: Create a string template for the chat assistant. It must indicate the LLM
        # that a chat history is being provided and that a new question is being asked.
        # The template must have two input variables: `history` and `human_input`.
        assistant_template = """
        The following is a friendly conversation between a human and an AI-agent. 
        The AI-agent is helpful, provides detailed information, and uses the chat history to maintain context.
        If the AI-agent does not know the answer to a question, it truthfully says it does not know.

        Chat history:
        {history}

        New question from the human:
        {human_input}

        AI-agent:
        """

        # TODO: Create a prompt template using the string template created above.
        # Hint: Use the `langchain.prompts.PromptTemplate` class.
        # Hint: Don't forget to add the input variables: `history` and `human_input`.
        self.prompt = PromptTemplate(
            input_variables=["history", "human_input"],
            template=assistant_template
        )

        # TODO: Create an instance of `langchain.chat_models.ChatOpenAI` with the appropriate settings.
        # Remember some settings are being provided in the __init__ function for this class.
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=api_key,
            temperature=temperature
        )

        # TODO: Create an instance of `langchain.chains.LLMChain` with the appropriate settings.
        # This chain must combine our prompt, llm and also have a memory.
        # Hint: You can use the `langchain.memory.ConversationBufferWindowMemory` class with
        # `k=history_length``.
        self.model = LLMChain(
            prompt=self.prompt,
            llm=self.llm,
            memory=ConversationBufferWindowMemory(k=history_length),
            verbose=settings.LANGCHAIN_VERBOSE 
        )

    def predict(self, human_input: str) -> str:
        """
        Generate a response to a human input.

        Parameters
        ----------
        human_input : str
            The human input to the chat assistant.

        Returns
        -------
        response : str
            The response from the chat assistant.
        """
        response = self.model.invoke(human_input)

        return response


if __name__ == "__main__":
    # Create an instance of ChatAssistant with appropriate settings
    chat_assistant = ChatAssistant(
        llm_model=settings.OPENAI_LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        history_length=2,
    )

    # Use the instance to generate a response
    output = chat_assistant.predict(
        human_input="what is the answer to life the universe and everything?"
    )

    print(output)

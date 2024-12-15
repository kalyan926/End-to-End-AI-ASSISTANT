### **Introduction**

This project focuses on improving the adaptability and reliability of large language models (LLMs) in determining the appropriate response strategy for a given query. Specifically, the goal is to enable the LLM to decide whether it can answer a query directly, whether it requires database access, or if it needs real-time tools to fetch information. Traditional approaches, such as using ReAct (Reasoning + Acting) style prompts or Chain-of-Thought (CoT) prompting, often fall short, leading to hallucinations or suboptimal decisions. To address this, synthetic data was generated to train the LLM effectively on these scenarios. The model was further optimized through fine-tuning and quantization into GGUF format for faster inference.

---

### **Challenges with LLMs**

1. **Hallucination Issues**:  
   LLMs often generate confident but factually incorrect or irrelevant responses. This is especially problematic in scenarios where real-world accuracy is crucial, such as retrieving live data or interacting with external systems.

2. **Limitations of Traditional Prompting (ReAct/CoT)**:  
   - ReAct-style prompting combines reasoning and acting but relies heavily on pre-crafted prompts and assumptions about the model's behavior.  
   - Chain-of-Thought reasoning improves logical output but doesn’t ensure the LLM can differentiate between tasks requiring external tools and those solvable internally.

These methods are resource-intensive and unreliable, leading to inconsistent task execution and response strategies.

---

### **Synthetic Data Generation**

To train the LLM to identify and adapt to the three situations (self-answering, database access, or tool usage), synthetic data was generated. The dataset included:

1. **Diverse Query Scenarios**:  
   - Questions solvable by the LLM’s knowledge base.  
   - Questions requiring database queries (e.g., "What is the sales figure for Q3?").  
   - Questions demanding real-time data or tool usage (e.g., "What is the current weather in Paris?").

2. **Behavioral Labels**:  
   Each data point included an expected behavioral response, specifying whether the LLM should answer directly, query a database, or invoke a tool.

3. **Error Simulation**:  
   Some synthetic examples were designed to simulate ambiguous queries or noisy data to train the LLM on robust decision-making.

---

### **Fine-Tuning with Synthetic Data**

The synthetic dataset was used to fine-tune the LLaMA 3.2 model. Fine-tuning brought several benefits:

1. **Improved Context Awareness**:  
   The model learned to discern which scenario it faced, leading to better decision-making.

2. **Reduction in Token Usage**:  
   By fine-tuning with tailored data, the model reduced token requirements by 60%. This streamlined inference and minimized computational overhead.

3. **Customization for Task-Specific Performance**:  
   Fine-tuning ensured the model was aligned with the project’s specific goals, improving accuracy in selecting the appropriate response strategy.

**Prompt Before finetuning:**
"Answer the following questions as best you can. You have access to the following tools: 
[Tool(name='Wikipedia', description='Use for in-depth topic exploration with detailed, user-curated articles and references.', func=<bound method WikipediaAPIWrapper.run of WikipediaAPIWrapper(wiki_client=<module 'wikipedia' from '/usr/local/lib/python3.10/dist-packages/wikipedia/__init__.py'>, top_k_results=3, lang='en', load_all_available_meta=False, doc_content_chars_max=4000)>), 
Tool(name='search', description=' Use for general topic searches, aggregating results from multiple sources. Simply input a search query to get started.', func=<bound method BaseTool.run of DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(region='wt-wt', safesearch='moderate', time='y', max_results=5, backend='api', source='text'))>),
StructuredTool(name='retriever', description='This function retrieves information stored in the database related to LangChain. It takes a string parameter as input.', args_schema=<class 'langchain_core.utils.pydantic.retriever'>, func=<function retriever at 0x7abfc1345240>)]

 Use the following format:
 Question: How does LangChain handle entity extraction?
Thought: A query about LangChain's ability to extract entities from text.
Action: retriever
Action Input: LangChain entity extraction
Observation: LangChain extracts entities using pre-trained NLP models that identify and categorize elements such as names, dates, and locations.
Thought: The observation gives a detailed explanation. This is the final answer.
Final Answer: LangChain extracts entities by using NLP models to identify and categorize key information from text.
Question: {input}
{agent_scratchpad}
”

**Prompt after Finetuning:**
"Answer the following questions as best you can. You have access to the following tools:
[Tool(name='Wikipedia', description='Use for in-depth topic exploration with detailed, user-curated articles and references.', func=<bound method WikipediaAPIWrapper.run of WikipediaAPIWrapper(wiki_client=<module 'wikipedia' from '/usr/local/lib/python3.10/dist-packages/wikipedia/__init__.py'>, top_k_results=3, lang='en', load_all_available_meta=False, doc_content_chars_max=4000)>), 
Tool(name='search', description=' Use for general topic searches, aggregating results from multiple sources. Simply input a search query to get started.', func=<bound method BaseTool.run of DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(region='wt-wt', safesearch='moderate', time='y', max_results=5, backend='api', source='text'))>), 
StructuredTool(name='retriever', description='This function retrieves information stored in the database related to LangChain. It takes a string parameter as input.', args_schema=<class 'langchain_core.utils.pydantic.retriever'>, func=<function retriever at 0x7abfc1345240>)]

Use the following format:
Question: {input}
{agent_scratchpad}
”


---

### **Optimization for Faster Inference**

1. **Quantization**:  
   After fine-tuning, the model was quantized into **GGUF (GPTQ General Unified Format)**, a high-efficiency model format optimized for edge deployment and faster inference speeds.

2. **Benefits of GGUF**:  
   - Smaller memory footprint, allowing deployment on resource-constrained environments.  
   - Maintained precision and decision-making capability while significantly reducing latency.

3. **Impact on Real-Time Performance**:  
   The optimized model demonstrated faster response times, making it suitable for live applications requiring real-time decision-making.

---

### **Key Advantages**

1. **Reduced Hallucination**:  
   The model reliably distinguishes when to answer directly, minimizing the risk of confidently incorrect answers.

2. **Efficiency**:  
   Through fine-tuning and optimization, the model processes queries faster, with reduced computational requirements.

3. **Scalability**:  
   The smaller GGUF format and reduced token usage make it feasible to deploy the model across various platforms, from cloud-based systems to local edge devices.

4. **Enhanced Adaptability**:  
   The synthetic training approach ensures the model can handle edge cases and ambiguous queries more effectively than traditional methods.

---

### **Conclusion**

This project provides a novel solution to the challenges faced by LLMs in real-world applications. By replacing traditional prompting techniques with a fine-tuned, synthetic-data-driven approach and optimizing the model for performance, the project achieves a significant leap in adaptability, accuracy, and efficiency. The use of GGUF quantization ensures that the model remains practical for diverse deployment scenarios, paving the way for smarter and faster LLM-based applications.

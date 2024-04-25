from llama_index.core.program import LLMTextCompletionProgram

class program:
    def __init__(self,outputclass,llm)->None:
        self.outputclass = outputclass
        self.llm = llm
    
    def programrespond(self,prompt)->LLMTextCompletionProgram:
        program = LLMTextCompletionProgram.from_defaults(
            output_cls = self.outputclass,
            llm = self.llm,
            prompt_template_str = prompt,
            verbose = True
            )
        return program

class program_advanced:
    def __init__(self,output_parser, outputclass, llm) -> None:
        self.outputclass = outputclass
        self.output_parser=output_parser
        self.llm = llm

    def programrespond(self, prompt_template)-> LLMTextCompletionProgram:
        program = LLMTextCompletionProgram.from_defaults(
            output_parser=self.output_parser(verbose=True),
            output_cls=self.outputclass,
            llm = self.llm,
            prompt_template_str= prompt_template,
            verbose=True,
        )

        return program